#!/usr/bin/env python3
"""Compare Dummy, multinomial L2, and elastic-net on locked development folds."""

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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

try:
    from evaluation_framework import build_pipeline, inner_splits, load_config
except ModuleNotFoundError:  # pragma: no cover - package import
    from scripts.evaluation_framework import build_pipeline, inner_splits, load_config


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CONFIG_PATH = ROOT / "config" / "evaluation.json"
COMPARISON_CONFIG_PATH = ROOT / "config" / "logistic_comparison_v1.json"
ASSIGNMENT_PATH = (
    ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv"
)
SPLIT_LOCK_PATH = ROOT / "data" / "processed" / "splits" / "split_lock.json"
GENE_PATH = ROOT / "data" / "processed" / "expression" / "genes.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "modeling" / "logistic_comparison"

MODEL_ORDER = [
    "dummy_prior",
    "multinomial_logistic_l2",
    "multinomial_logistic_elastic_net",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_split_lock() -> dict[str, Any]:
    lock = json.loads(SPLIT_LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("test_set_model_accessed") is not False:
        raise RuntimeError("Split lock does not certify an untouched locked test set")
    for relative_path, expected in lock["artifacts_sha256"].items():
        observed = sha256(ROOT / relative_path)
        if observed != expected:
            raise RuntimeError(f"Locked artifact hash mismatch: {relative_path}")
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def safe_label(label: str) -> str:
    return label.lower().replace("-", "_").replace(" ", "_")


def make_search(
    model_name: str,
    evaluation_config: dict[str, Any],
    comparison_config: dict[str, Any],
    cv: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    n_jobs: int,
) -> GridSearchCV:
    base_name = "dummy_prior" if model_name == "dummy_prior" else "multinomial_logistic"
    pipeline = build_pipeline(base_name, evaluation_config, random_seed=seed)
    pipeline.set_params(variance_selector__k=1000)
    model_config = comparison_config["models"][model_name]
    grid = dict(model_config["parameter_grid"])
    if not grid:
        grid = {"variance_selector__k": [1000]}
    return GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring={
            "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
        },
        refit=comparison_config["inner_selection_metric"],
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )


def selected_matrix_columns(
    best_pipeline: Any, protein_coding_indices: np.ndarray
) -> np.ndarray:
    low_indices = best_pipeline.named_steps["low_expression_filter"].get_support(
        indices=True
    )
    variance_indices = best_pipeline.named_steps["variance_selector"].get_support(
        indices=True
    )
    return protein_coding_indices[low_indices[variance_indices]]


def coefficient_records(
    *,
    model_name: str,
    outer_fold: int,
    best_pipeline: Any,
    protein_coding_indices: np.ndarray,
    genes: pd.DataFrame,
    threshold: float,
) -> tuple[list[dict[str, Any]], int, int]:
    classifier = best_pipeline.named_steps["classifier"]
    columns = selected_matrix_columns(best_pipeline, protein_coding_indices)
    selected_genes = genes.set_index("matrix_column").loc[columns].reset_index()
    coefficients = np.asarray(classifier.coef_)
    if coefficients.shape[1] != len(selected_genes):
        raise RuntimeError("Coefficient and selected-gene axes do not align")
    records: list[dict[str, Any]] = []
    nonzero_mask = np.abs(coefficients) > threshold
    for class_index, subtype in enumerate(classifier.classes_):
        for gene_index, gene in selected_genes.iterrows():
            value = float(coefficients[class_index, gene_index])
            records.append(
                {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "pam50_class": subtype,
                    "matrix_column": int(gene["matrix_column"]),
                    "gene_id": gene["gene_id"],
                    "gene_id_without_version": gene["gene_id_without_version"],
                    "gene_name": gene["gene_name"],
                    "coefficient": value,
                    "absolute_coefficient": abs(value),
                    "is_nonzero": bool(nonzero_mask[class_index, gene_index]),
                }
            )
    unique_nonzero = int(nonzero_mask.any(axis=0).sum())
    total_nonzero = int(nonzero_mask.sum())
    return records, unique_nonzero, total_nonzero


def main() -> None:
    args = parse_args()
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero")
    split_lock = verify_split_lock()
    evaluation_config = load_config(EVALUATION_CONFIG_PATH)
    comparison_config = load_config(COMPARISON_CONFIG_PATH)
    seed = int(comparison_config["random_seed"])
    assignments = pd.read_csv(ASSIGNMENT_PATH, sep="\t")
    locked_test = assignments.loc[assignments["holdout_split"].eq("locked_test")]
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    if len(development) != 756 or len(locked_test) != 189:
        raise RuntimeError("Unexpected four-class partition sizes")
    if locked_test["outer_fold"].notna().any():
        raise RuntimeError("Locked-test rows unexpectedly have CV fold assignments")
    if development["patient_group"].duplicated().any():
        raise RuntimeError("Current case-level development cohort must have unique patients")

    matrix_path = ROOT / evaluation_config["features"]["matrix"]
    coding_path = ROOT / evaluation_config["features"]["candidate_gene_indices"]
    matrix = np.load(matrix_path, mmap_mode="r")
    protein_coding_indices = np.load(coding_path)
    development_rows = development["matrix_row"].astype(int).to_numpy()
    if np.intersect1d(development_rows, locked_test["matrix_row"].astype(int)).size:
        raise RuntimeError("Development and locked-test matrix rows overlap")
    X = np.asarray(matrix[development_rows][:, protein_coding_indices], dtype=np.float32)
    y = development[evaluation_config["label_column"]].astype(str).to_numpy()
    genes = pd.read_csv(GENE_PATH, sep="\t")
    class_order = list(evaluation_config["primary_analysis"]["classes"])
    threshold = float(comparison_config["coefficient_nonzero_threshold"])

    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    best_parameters: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    warning_records: list[dict[str, Any]] = []
    outer_values = development["outer_fold"].astype(int).to_numpy()

    for model_index, model_name in enumerate(MODEL_ORDER):
        for outer_fold in range(1, 6):
            print(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] "
                f"{model_name}, outer fold {outer_fold}/5",
                flush=True,
            )
            outer_valid_index = np.flatnonzero(outer_values == outer_fold)
            outer_train_index = np.flatnonzero(outer_values != outer_fold)
            cv = inner_splits(development, outer_fold)
            search = make_search(
                model_name,
                evaluation_config,
                comparison_config,
                cv,
                seed=seed + 10000 * (model_index + 1) + outer_fold,
                n_jobs=args.n_jobs,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                search.fit(X[outer_train_index], y[outer_train_index])
            for warning in caught:
                if issubclass(warning.category, ConvergenceWarning):
                    warning_records.append(
                        {
                            "model": model_name,
                            "outer_fold": outer_fold,
                            "category": warning.category.__name__,
                            "message": str(warning.message),
                        }
                    )

            best = search.best_estimator_
            predicted = best.predict(X[outer_valid_index])
            probabilities = best.predict_proba(X[outer_valid_index])
            classifier = best.named_steps["classifier"]
            probability_frame = pd.DataFrame(
                probabilities,
                columns=[safe_label(str(label)) for label in classifier.classes_],
            ).reindex(columns=[safe_label(label) for label in class_order])
            truth = y[outer_valid_index]
            unique_nonzero: int | None = None
            total_nonzero: int | None = None
            if model_name != "dummy_prior":
                fold_coefficients, unique_nonzero, total_nonzero = coefficient_records(
                    model_name=model_name,
                    outer_fold=outer_fold,
                    best_pipeline=best,
                    protein_coding_indices=protein_coding_indices,
                    genes=genes,
                    threshold=threshold,
                )
                coefficients.extend(fold_coefficients)

            low_count = int(
                best.named_steps["low_expression_filter"].get_support().sum()
            )
            metrics.append(
                {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "outer_train_n": int(len(outer_train_index)),
                    "outer_validation_n": int(len(outer_valid_index)),
                    "low_expression_pass_genes": low_count,
                    "selected_genes": int(
                        best.named_steps["variance_selector"].get_support().sum()
                    ),
                    "unique_nonzero_genes": unique_nonzero,
                    "total_nonzero_class_coefficients": total_nonzero,
                    "macro_f1": float(f1_score(truth, predicted, average="macro")),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(truth, predicted)
                    ),
                    "accuracy": float(accuracy_score(truth, predicted)),
                    "macro_ovr_roc_auc": float(
                        roc_auc_score(
                            truth,
                            probabilities,
                            labels=classifier.classes_,
                            multi_class="ovr",
                            average="macro",
                        )
                    ),
                    "multiclass_log_loss": float(
                        log_loss(
                            truth, probabilities, labels=classifier.classes_
                        )
                    ),
                    "inner_best_macro_f1": float(search.best_score_),
                    "convergence_warning_count": int(
                        sum(
                            issubclass(w.category, ConvergenceWarning) for w in caught
                        )
                    ),
                }
            )
            best_parameters.append(
                {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "best_inner_macro_f1": float(search.best_score_),
                    "best_parameters_json": json.dumps(
                        search.best_params_, sort_keys=True
                    ),
                    "class_weight": getattr(classifier, "class_weight", None),
                    "strategy": getattr(classifier, "strategy", None),
                    "max_n_iter": (
                        int(np.max(classifier.n_iter_))
                        if hasattr(classifier, "n_iter_")
                        else None
                    ),
                }
            )
            for position, local_index in enumerate(outer_valid_index):
                sample = development.iloc[int(local_index)]
                record: dict[str, Any] = {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "matrix_row": int(sample["matrix_row"]),
                    "case_barcode": sample["case_barcode"],
                    "sample_barcode": sample["sample_barcode"],
                    "observed": truth[position],
                    "predicted": predicted[position],
                }
                for label in class_order:
                    record[f"probability_{safe_label(label)}"] = float(
                        probability_frame.iloc[position][safe_label(label)]
                    )
                predictions.append(record)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics)
    predictions_frame = pd.DataFrame(predictions)
    parameters_frame = pd.DataFrame(best_parameters)
    coefficients_frame = pd.DataFrame(coefficients)
    warnings_frame = pd.DataFrame(warning_records)
    metrics_frame.to_csv(output_dir / "outer_fold_metrics.tsv", sep="\t", index=False)
    predictions_frame.to_csv(
        output_dir / "outer_fold_predictions.tsv", sep="\t", index=False
    )
    parameters_frame.to_csv(
        output_dir / "best_hyperparameters.tsv", sep="\t", index=False
    )
    coefficients_frame.to_csv(
        output_dir / "outer_fold_coefficients.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    warnings_frame.to_csv(
        output_dir / "convergence_warnings.tsv", sep="\t", index=False
    )

    metric_columns = [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "macro_ovr_roc_auc",
        "multiclass_log_loss",
        "unique_nonzero_genes",
    ]
    summary = metrics_frame.groupby("model", sort=False)[metric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "model_summary.tsv", sep="\t", index=False)

    paired_records: list[dict[str, Any]] = []
    pivot = metrics_frame.pivot(index="outer_fold", columns="model")
    for comparator in [
        "multinomial_logistic_l2",
        "multinomial_logistic_elastic_net",
    ]:
        baseline = "dummy_prior" if comparator.endswith("l2") else "multinomial_logistic_l2"
        for metric in ["macro_f1", "balanced_accuracy", "macro_ovr_roc_auc"]:
            differences = pivot[metric][comparator] - pivot[metric][baseline]
            for outer_fold, difference in differences.items():
                paired_records.append(
                    {
                        "comparison": f"{comparator}_minus_{baseline}",
                        "metric": metric,
                        "outer_fold": int(outer_fold),
                        "difference": float(difference),
                    }
                )
    pd.DataFrame(paired_records).to_csv(
        output_dir / "paired_outer_fold_differences.tsv", sep="\t", index=False
    )

    run_manifest = {
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort": "four_class",
        "development_n": int(len(development)),
        "locked_test_n": int(len(locked_test)),
        "locked_test_expression_rows_loaded": 0,
        "locked_test_predictions_generated": 0,
        "models": MODEL_ORDER,
        "logistic_class_weight": "balanced",
        "outer_folds": 5,
        "inner_folds": 5,
        "random_seed": seed,
        "split_lock_sha256": sha256(SPLIT_LOCK_PATH),
        "evaluation_config_sha256": sha256(EVALUATION_CONFIG_PATH),
        "comparison_config_sha256": sha256(COMPARISON_CONFIG_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "matrix_manifest_sha256": split_lock["artifacts_sha256"][
            "data/processed/expression/matrix_manifest.json"
        ],
    }
    artifact_paths = [
        "outer_fold_metrics.tsv",
        "outer_fold_predictions.tsv",
        "best_hyperparameters.tsv",
        "outer_fold_coefficients.tsv.gz",
        "convergence_warnings.tsv",
        "model_summary.tsv",
        "paired_outer_fold_differences.tsv",
    ]
    run_manifest["output_sha256"] = {
        name: sha256(output_dir / name) for name in artifact_paths
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
