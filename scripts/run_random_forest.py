#!/usr/bin/env python3
"""Run nested-CV RandomForest with training-fold feature selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

try:
    from evaluation_framework import build_pipeline, inner_splits, load_config
except ModuleNotFoundError:  # pragma: no cover - package import
    from scripts.evaluation_framework import build_pipeline, inner_splits, load_config


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CONFIG_PATH = ROOT / "config" / "evaluation.json"
FOREST_CONFIG_PATH = ROOT / "config" / "random_forest_v1.json"
ASSIGNMENT_PATH = (
    ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv"
)
SPLIT_LOCK_PATH = ROOT / "data" / "processed" / "splits" / "split_lock.json"
GENE_PATH = ROOT / "data" / "processed" / "expression" / "genes.tsv"
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "random_forest"
CLASS_ORDER = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_split_lock() -> dict[str, Any]:
    lock = json.loads(SPLIT_LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("test_set_model_accessed") is not False:
        raise RuntimeError("Split lock does not certify an untouched test set")
    for relative_path, expected in lock["artifacts_sha256"].items():
        if sha256(ROOT / relative_path) != expected:
            raise RuntimeError(f"Locked artifact hash mismatch: {relative_path}")
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forest-n-jobs",
        type=int,
        default=None,
        help="Override the locked per-forest thread count; use only for environment compatibility.",
    )
    return parser.parse_args()


def safe_label(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def sklearn_parameter_grid(config: dict[str, Any]) -> list[dict[str, list[Any]]]:
    return [
        {key: [value] for key, value in candidate.items()}
        for candidate in config["classifier"]["candidate_parameter_sets"]
    ]


def selected_gene_frame(
    pipeline: Any, protein_coding_indices: np.ndarray, genes: pd.DataFrame
) -> pd.DataFrame:
    low_indices = pipeline.named_steps["low_expression_filter"].get_support(indices=True)
    variance_indices = pipeline.named_steps["variance_selector"].get_support(indices=True)
    matrix_columns = protein_coding_indices[low_indices[variance_indices]]
    return genes.set_index("matrix_column").loc[matrix_columns].reset_index()


def main() -> None:
    args = parse_args()
    split_lock = verify_split_lock()
    evaluation_config = load_config(EVALUATION_CONFIG_PATH)
    forest_config = load_config(FOREST_CONFIG_PATH)
    if forest_config["classifier"]["class"] != "RandomForestClassifier":
        raise RuntimeError("Unexpected classifier configuration")
    if forest_config["classifier"]["class_weight"] != "balanced":
        raise RuntimeError("Random forest must use class_weight='balanced'")
    forest_n_jobs = (
        int(args.forest_n_jobs)
        if args.forest_n_jobs is not None
        else int(forest_config["classifier"]["forest_n_jobs"])
    )
    if forest_n_jobs == 0:
        raise ValueError("forest_n_jobs cannot be zero")

    assignments = pd.read_csv(ASSIGNMENT_PATH, sep="\t")
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    locked_test = assignments.loc[assignments["holdout_split"].eq("locked_test")]
    if len(development) != 756 or len(locked_test) != 189:
        raise RuntimeError("Unexpected locked partition sizes")
    development_rows = development["matrix_row"].astype(int).to_numpy()
    locked_rows = locked_test["matrix_row"].astype(int).to_numpy()
    if np.intersect1d(development_rows, locked_rows).size:
        raise RuntimeError("Development and locked-test matrix rows overlap")
    if locked_test["outer_fold"].notna().any():
        raise RuntimeError("Locked-test rows must not have CV fold assignments")

    matrix = np.load(
        ROOT / evaluation_config["features"]["matrix"], mmap_mode="r"
    )
    protein_coding_indices = np.load(
        ROOT / evaluation_config["features"]["candidate_gene_indices"]
    )
    X = np.asarray(matrix[development_rows][:, protein_coding_indices], dtype=np.float32)
    y = development[evaluation_config["label_column"]].astype(str).to_numpy()
    genes = pd.read_csv(GENE_PATH, sep="\t")
    seed = int(forest_config["random_seed"])
    outer_values = development["outer_fold"].astype(int).to_numpy()
    parameter_grid = sklearn_parameter_grid(forest_config)

    metric_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    parameter_records: list[dict[str, Any]] = []
    importance_records: list[dict[str, Any]] = []
    warning_records: list[dict[str, Any]] = []

    for outer_fold in range(1, 6):
        print(
            f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] "
            f"RandomForest outer fold {outer_fold}/5",
            flush=True,
        )
        outer_valid_index = np.flatnonzero(outer_values == outer_fold)
        outer_train_index = np.flatnonzero(outer_values != outer_fold)
        cv = inner_splits(development, outer_fold)
        pipeline = build_pipeline(
            "random_forest", evaluation_config, random_seed=seed + outer_fold
        )
        pipeline.set_params(
            classifier__class_weight="balanced",
            classifier__criterion="gini",
            classifier__bootstrap=True,
            classifier__n_jobs=forest_n_jobs,
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring={
                "macro_f1": "f1_macro",
                "balanced_accuracy": "balanced_accuracy",
            },
            refit=forest_config["inner_selection_metric"],
            cv=cv,
            n_jobs=int(forest_config["classifier"]["grid_search_n_jobs"]),
            return_train_score=False,
            error_score="raise",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(X[outer_train_index], y[outer_train_index])
        for warning in caught:
            warning_records.append(
                {
                    "outer_fold": outer_fold,
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                }
            )

        best = search.best_estimator_
        classifier = best.named_steps["classifier"]
        if classifier.__class__.__name__ != "RandomForestClassifier":
            raise RuntimeError("Unexpected fitted classifier")
        if classifier.class_weight != "balanced":
            raise RuntimeError("Fitted classifier lost balanced class weights")
        predicted = best.predict(X[outer_valid_index])
        probabilities = best.predict_proba(X[outer_valid_index])
        truth = y[outer_valid_index]
        selected_genes = selected_gene_frame(best, protein_coding_indices, genes)
        if len(selected_genes) != len(classifier.feature_importances_):
            raise RuntimeError("Selected-gene and importance axes do not align")
        nonzero_importance = classifier.feature_importances_ > 0
        for gene_index, gene in selected_genes.iterrows():
            importance_records.append(
                {
                    "outer_fold": outer_fold,
                    "matrix_column": int(gene["matrix_column"]),
                    "gene_id": gene["gene_id"],
                    "gene_id_without_version": gene["gene_id_without_version"],
                    "gene_name": gene["gene_name"],
                    "impurity_importance": float(
                        classifier.feature_importances_[gene_index]
                    ),
                    "nonzero_impurity_importance": bool(
                        nonzero_importance[gene_index]
                    ),
                }
            )

        probability_index = {
            label: index for index, label in enumerate(classifier.classes_)
        }
        probabilities_in_order = np.column_stack(
            [probabilities[:, probability_index[label]] for label in classifier.classes_]
        )
        metric_records.append(
            {
                "model": "random_forest",
                "outer_fold": outer_fold,
                "outer_train_n": int(len(outer_train_index)),
                "outer_validation_n": int(len(outer_valid_index)),
                "low_expression_pass_genes": int(
                    best.named_steps["low_expression_filter"].get_support().sum()
                ),
                "selected_genes": int(len(selected_genes)),
                "nonzero_impurity_importance_genes": int(nonzero_importance.sum()),
                "macro_f1": float(f1_score(truth, predicted, average="macro")),
                "balanced_accuracy": float(
                    balanced_accuracy_score(truth, predicted)
                ),
                "accuracy": float(accuracy_score(truth, predicted)),
                "macro_ovr_roc_auc": float(
                    roc_auc_score(
                        truth,
                        probabilities_in_order,
                        labels=classifier.classes_,
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                "multiclass_log_loss": float(
                    log_loss(
                        truth,
                        probabilities_in_order,
                        labels=classifier.classes_,
                    )
                ),
                "inner_best_macro_f1": float(search.best_score_),
                "warning_count": int(len(caught)),
            }
        )
        parameter_records.append(
            {
                "model": "random_forest",
                "outer_fold": outer_fold,
                "best_inner_macro_f1": float(search.best_score_),
                "best_parameters_json": json.dumps(search.best_params_, sort_keys=True),
                "class_weight": classifier.class_weight,
                "classifier_class": classifier.__class__.__name__,
                "n_jobs": classifier.n_jobs,
            }
        )
        for position, local_index in enumerate(outer_valid_index):
            sample = development.iloc[int(local_index)]
            record: dict[str, Any] = {
                "model": "random_forest",
                "outer_fold": outer_fold,
                "matrix_row": int(sample["matrix_row"]),
                "case_barcode": sample["case_barcode"],
                "sample_barcode": sample["sample_barcode"],
                "observed": truth[position],
                "predicted": predicted[position],
            }
            for label in CLASS_ORDER:
                record[f"probability_{safe_label(label)}"] = float(
                    probabilities[position, probability_index[label]]
                )
            prediction_records.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_records)
    predictions = pd.DataFrame(prediction_records)
    parameters = pd.DataFrame(parameter_records)
    importances = pd.DataFrame(importance_records)
    warning_frame = pd.DataFrame(warning_records)
    metrics.to_csv(OUTPUT_DIR / "outer_fold_metrics.tsv", sep="\t", index=False)
    predictions.to_csv(
        OUTPUT_DIR / "outer_fold_predictions.tsv", sep="\t", index=False
    )
    parameters.to_csv(
        OUTPUT_DIR / "best_hyperparameters.tsv", sep="\t", index=False
    )
    importances.to_csv(
        OUTPUT_DIR / "outer_fold_impurity_importance.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    warning_frame.to_csv(OUTPUT_DIR / "warnings.tsv", sep="\t", index=False)

    summary_columns = [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "macro_ovr_roc_auc",
        "multiclass_log_loss",
        "selected_genes",
        "nonzero_impurity_importance_genes",
    ]
    summary_values: dict[str, Any] = {"model": "random_forest"}
    for column in summary_columns:
        summary_values[f"{column}_mean"] = float(metrics[column].mean())
        summary_values[f"{column}_std"] = float(metrics[column].std(ddof=1))
    summary = pd.DataFrame([summary_values])
    summary.to_csv(OUTPUT_DIR / "model_summary.tsv", sep="\t", index=False)

    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["observed"],
        predictions["predicted"],
        labels=CLASS_ORDER,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "model": "random_forest",
            "pam50_class": CLASS_ORDER,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    per_class.to_csv(
        OUTPUT_DIR / "oof_per_class_metrics.tsv", sep="\t", index=False
    )

    manifest = {
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "RandomForestClassifier",
        "class_weight": "balanced",
        "development_n": 756,
        "locked_test_n": 189,
        "locked_test_expression_rows_loaded": 0,
        "locked_test_predictions_generated": 0,
        "outer_folds": 5,
        "inner_folds": 5,
        "candidate_parameter_sets": len(parameter_grid),
        "random_seed": seed,
        "forest_n_jobs": forest_n_jobs,
        "split_lock_sha256": sha256(SPLIT_LOCK_PATH),
        "evaluation_config_sha256": sha256(EVALUATION_CONFIG_PATH),
        "random_forest_config_sha256": sha256(FOREST_CONFIG_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "matrix_manifest_sha256": split_lock["artifacts_sha256"][
            "data/processed/expression/matrix_manifest.json"
        ],
        "importance_caveat": forest_config["importance_policy"],
    }
    artifact_names = [
        "outer_fold_metrics.tsv",
        "outer_fold_predictions.tsv",
        "best_hyperparameters.tsv",
        "outer_fold_impurity_importance.tsv.gz",
        "warnings.tsv",
        "model_summary.tsv",
        "oof_per_class_metrics.tsv",
    ]
    manifest["output_sha256"] = {
        name: sha256(OUTPUT_DIR / name) for name in artifact_names
    }
    with (OUTPUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
