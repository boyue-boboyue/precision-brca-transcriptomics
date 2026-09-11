#!/usr/bin/env python3
"""Run all locked candidate models after excluding the PAM50 signature genes."""

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
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize

try:
    from evaluation_framework import build_pipeline, inner_splits, load_config
except ModuleNotFoundError:  # pragma: no cover
    from scripts.evaluation_framework import build_pipeline, inner_splits, load_config


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CONFIG_PATH = ROOT / "config" / "evaluation.json"
LOGISTIC_CONFIG_PATH = ROOT / "config" / "logistic_comparison_v1.json"
SVC_CONFIG_PATH = ROOT / "config" / "linear_svc_v1.json"
FOREST_CONFIG_PATH = ROOT / "config" / "random_forest_v1.json"
EXCLUSION_CONFIG_PATH = ROOT / "config" / "pam50_exclusion_v1.json"
PAM50_GENES_PATH = (
    ROOT / "data" / "processed" / "labels" / "pam50_signature_genes_v1.tsv"
)
ASSIGNMENT_PATH = (
    ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv"
)
SPLIT_LOCK_PATH = ROOT / "data" / "processed" / "splits" / "split_lock.json"
GENE_PATH = ROOT / "data" / "processed" / "expression" / "genes.tsv"
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "pam50_excluded"
CLASS_ORDER = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]
MODEL_ORDER = [
    "dummy_prior",
    "multinomial_logistic_l2",
    "multinomial_logistic_elastic_net",
    "linear_svc",
    "random_forest",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_label(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def verify_locks() -> dict[str, Any]:
    split_lock = json.loads(SPLIT_LOCK_PATH.read_text(encoding="utf-8"))
    if split_lock.get("test_set_model_accessed") is not False:
        raise RuntimeError("Split lock does not certify an untouched test set")
    for relative_path, expected in split_lock["artifacts_sha256"].items():
        if sha256(ROOT / relative_path) != expected:
            raise RuntimeError(f"Locked artifact hash mismatch: {relative_path}")

    exclusion_lock = json.loads(EXCLUSION_CONFIG_PATH.read_text(encoding="utf-8"))
    if sha256(PAM50_GENES_PATH) != exclusion_lock["locked_gene_table_sha256"]:
        raise RuntimeError("PAM50 signature table hash mismatch")
    for relative_path, expected in exclusion_lock["inputs_sha256"].items():
        if sha256(ROOT / relative_path) != expected:
            raise RuntimeError(f"PAM50 exclusion input hash mismatch: {relative_path}")
    return split_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--forest-n-jobs", type=int, default=4)
    return parser.parse_args()


def model_search(
    model_name: str,
    evaluation_config: dict[str, Any],
    logistic_config: dict[str, Any],
    svc_config: dict[str, Any],
    forest_config: dict[str, Any],
    cv: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_jobs: int,
    forest_n_jobs: int,
) -> GridSearchCV:
    if model_name in {
        "dummy_prior",
        "multinomial_logistic_l2",
        "multinomial_logistic_elastic_net",
    }:
        base_name = (
            "dummy_prior" if model_name == "dummy_prior" else "multinomial_logistic"
        )
        pipeline = build_pipeline(base_name, evaluation_config, random_seed=seed)
        pipeline.set_params(variance_selector__k=1000)
        grid = dict(logistic_config["models"][model_name]["parameter_grid"])
        if not grid:
            grid = {"variance_selector__k": [1000]}
    elif model_name == "linear_svc":
        pipeline = build_pipeline("linear_svm", evaluation_config, random_seed=seed)
        pipeline.set_params(variance_selector__k=1000)
        grid = dict(svc_config["classifier"]["parameter_grid"])
    elif model_name == "random_forest":
        pipeline = build_pipeline("random_forest", evaluation_config, random_seed=seed)
        pipeline.set_params(
            classifier__class_weight="balanced",
            classifier__criterion="gini",
            classifier__bootstrap=True,
            classifier__n_jobs=forest_n_jobs,
        )
        grid = [
            {key: [value] for key, value in candidate.items()}
            for candidate in forest_config["classifier"]["candidate_parameter_sets"]
        ]
    else:  # pragma: no cover
        raise ValueError(model_name)
    return GridSearchCV(
        pipeline,
        grid,
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
        refit="macro_f1",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )


def selected_matrix_columns(
    pipeline: Any, candidate_indices: np.ndarray
) -> np.ndarray:
    low = pipeline.named_steps["low_expression_filter"].get_support(indices=True)
    variance = pipeline.named_steps["variance_selector"].get_support(indices=True)
    return candidate_indices[low[variance]]


def probability_order(
    probabilities: np.ndarray, fitted_classes: np.ndarray
) -> np.ndarray:
    positions = {str(label): index for index, label in enumerate(fitted_classes)}
    return np.column_stack([probabilities[:, positions[label]] for label in CLASS_ORDER])


def ordered_multiclass_log_loss(truth: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute log loss for probabilities stored in the explicit CLASS_ORDER."""

    positions = {label: index for index, label in enumerate(CLASS_ORDER)}
    true_positions = np.asarray([positions[str(label)] for label in truth], dtype=int)
    selected = probabilities[np.arange(len(truth)), true_positions]
    return float(-np.log(np.clip(selected, 1e-15, 1.0)).mean())


def main() -> None:
    args = parse_args()
    if args.n_jobs == 0 or args.forest_n_jobs == 0:
        raise ValueError("Parallel job counts cannot be zero")
    split_lock = verify_locks()
    evaluation_config = load_config(EVALUATION_CONFIG_PATH)
    logistic_config = load_config(LOGISTIC_CONFIG_PATH)
    svc_config = load_config(SVC_CONFIG_PATH)
    forest_config = load_config(FOREST_CONFIG_PATH)

    assignments = pd.read_csv(ASSIGNMENT_PATH, sep="\t")
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    locked_test = assignments.loc[assignments["holdout_split"].eq("locked_test")]
    if len(development) != 756 or len(locked_test) != 189:
        raise RuntimeError("Unexpected four-class partition sizes")
    if locked_test["outer_fold"].notna().any():
        raise RuntimeError("Locked test must not have CV fold assignments")

    protein_coding = np.load(
        ROOT / evaluation_config["features"]["candidate_gene_indices"]
    ).astype(int)
    pam50 = pd.read_csv(PAM50_GENES_PATH, sep="\t")
    excluded_columns = pam50["matrix_column"].astype(int).to_numpy()
    if not np.isin(excluded_columns, protein_coding).all():
        raise RuntimeError("Every PAM50 gene must be on the protein-coding axis")
    candidate_indices = protein_coding[~np.isin(protein_coding, excluded_columns)]
    if len(candidate_indices) != len(protein_coding) - 50:
        raise RuntimeError("Expected exactly 50 genes to be excluded")

    matrix = np.load(ROOT / evaluation_config["features"]["matrix"], mmap_mode="r")
    development_rows = development["matrix_row"].astype(int).to_numpy()
    locked_rows = locked_test["matrix_row"].astype(int).to_numpy()
    if np.intersect1d(development_rows, locked_rows).size:
        raise RuntimeError("Development and locked-test rows overlap")
    # Only development rows are materialized. Locked-test expression is not loaded.
    X = np.asarray(matrix[development_rows][:, candidate_indices], dtype=np.float32)
    y = development[evaluation_config["label_column"]].astype(str).to_numpy()
    genes = pd.read_csv(GENE_PATH, sep="\t").set_index("matrix_column")
    outer_values = development["outer_fold"].astype(int).to_numpy()
    seed = int(evaluation_config["random_seed"])

    metric_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    parameter_records: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    warning_records: list[dict[str, Any]] = []

    for model_index, model_name in enumerate(MODEL_ORDER):
        for outer_fold in range(1, 6):
            print(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] "
                f"PAM50-excluded {model_name}, outer fold {outer_fold}/5",
                flush=True,
            )
            valid_index = np.flatnonzero(outer_values == outer_fold)
            train_index = np.flatnonzero(outer_values != outer_fold)
            cv = inner_splits(development, outer_fold)
            search = model_search(
                model_name,
                evaluation_config,
                logistic_config,
                svc_config,
                forest_config,
                cv,
                seed + 10000 * (model_index + 1) + outer_fold,
                args.n_jobs,
                args.forest_n_jobs,
            )
            with warnings.catch_warnings(record=True) as caught_search:
                warnings.simplefilter("always", ConvergenceWarning)
                search.fit(X[train_index], y[train_index])

            best = search.best_estimator_
            classifier = best.named_steps["classifier"]
            predicted = best.predict(X[valid_index])
            truth = y[valid_index]
            calibrated_predicted: np.ndarray | None = None
            decision_scores: np.ndarray | None = None
            caught_calibration: list[Any] = []

            if model_name == "linear_svc":
                decision_scores = best.decision_function(X[valid_index])
                calibrator = CalibratedClassifierCV(
                    estimator=clone(best),
                    method=svc_config["probability_calibration"]["method"],
                    cv=cv,
                    n_jobs=1,
                    ensemble=svc_config["probability_calibration"]["ensemble"],
                )
                with warnings.catch_warnings(record=True) as caught_calibration_raw:
                    warnings.simplefilter("always", ConvergenceWarning)
                    calibrator.fit(X[train_index], y[train_index])
                caught_calibration = list(caught_calibration_raw)
                probabilities = probability_order(
                    calibrator.predict_proba(X[valid_index]), calibrator.classes_
                )
                calibrated_predicted = calibrator.predict(X[valid_index])
                decision_scores = probability_order(decision_scores, classifier.classes_)
            else:
                probabilities = probability_order(
                    best.predict_proba(X[valid_index]), classifier.classes_
                )

            selected_columns = selected_matrix_columns(best, candidate_indices)
            overlap = np.intersect1d(selected_columns, excluded_columns)
            if overlap.size:
                raise RuntimeError("PAM50 gene entered a PAM50-excluded fitted pipeline")
            for matrix_column in selected_columns:
                gene = genes.loc[int(matrix_column)]
                feature_records.append(
                    {
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "matrix_column": int(matrix_column),
                        "gene_id": gene["gene_id"],
                        "gene_name": gene["gene_name"],
                        "is_locked_pam50_gene": False,
                    }
                )

            all_warnings = list(caught_search) + caught_calibration
            for phase, caught in (
                ("inner_search", list(caught_search)),
                ("training_only_calibration", caught_calibration),
            ):
                for warning in caught:
                    warning_records.append(
                        {
                            "model": model_name,
                            "outer_fold": outer_fold,
                            "phase": phase,
                            "category": warning.category.__name__,
                            "message": str(warning.message),
                        }
                    )

            binary_truth = label_binarize(truth, classes=CLASS_ORDER)
            metric_records.append(
                {
                    "feature_scheme": "pam50_excluded",
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "outer_train_n": int(len(train_index)),
                    "outer_validation_n": int(len(valid_index)),
                    "candidate_genes_before_filter": int(len(candidate_indices)),
                    "pam50_genes_removed_before_pipeline": 50,
                    "low_expression_pass_genes": int(
                        best.named_steps["low_expression_filter"].get_support().sum()
                    ),
                    "selected_genes": int(len(selected_columns)),
                    "selected_pam50_overlap": int(len(overlap)),
                    "macro_f1": float(f1_score(truth, predicted, average="macro")),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(truth, predicted)
                    ),
                    "accuracy": float(accuracy_score(truth, predicted)),
                    "macro_ovr_roc_auc": float(
                        roc_auc_score(binary_truth, probabilities, average="macro")
                    ),
                    "macro_average_precision": float(
                        average_precision_score(binary_truth, probabilities, average="macro")
                    ),
                    "multiclass_log_loss": float(
                        ordered_multiclass_log_loss(truth, probabilities)
                    ),
                    "inner_best_macro_f1": float(search.best_score_),
                    "warning_count": int(len(all_warnings)),
                }
            )
            parameter_records.append(
                {
                    "feature_scheme": "pam50_excluded",
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "best_inner_macro_f1": float(search.best_score_),
                    "best_parameters_json": json.dumps(
                        search.best_params_, sort_keys=True
                    ),
                    "classifier_class": classifier.__class__.__name__,
                    "class_weight": getattr(classifier, "class_weight", None),
                    "probability_source": (
                        "training_only_sigmoid_calibration"
                        if model_name == "linear_svc"
                        else "native_predict_proba"
                    ),
                }
            )
            for position, local_index in enumerate(valid_index):
                sample = development.iloc[int(local_index)]
                record: dict[str, Any] = {
                    "feature_scheme": "pam50_excluded",
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "matrix_row": int(sample["matrix_row"]),
                    "case_barcode": sample["case_barcode"],
                    "sample_barcode": sample["sample_barcode"],
                    "observed": truth[position],
                    "predicted": predicted[position],
                    "calibrated_predicted": (
                        calibrated_predicted[position]
                        if calibrated_predicted is not None
                        else predicted[position]
                    ),
                }
                for class_index, label in enumerate(CLASS_ORDER):
                    record[f"probability_{safe_label(label)}"] = float(
                        probabilities[position, class_index]
                    )
                    if decision_scores is not None:
                        record[f"decision_{safe_label(label)}"] = float(
                            decision_scores[position, class_index]
                        )
                prediction_records.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_records)
    predictions = pd.DataFrame(prediction_records)
    parameters = pd.DataFrame(parameter_records)
    features = pd.DataFrame(feature_records)
    warnings_frame = pd.DataFrame(warning_records)
    metrics.to_csv(OUTPUT_DIR / "outer_fold_metrics.tsv", sep="\t", index=False)
    predictions.to_csv(
        OUTPUT_DIR / "outer_fold_predictions.tsv", sep="\t", index=False
    )
    parameters.to_csv(
        OUTPUT_DIR / "best_hyperparameters.tsv", sep="\t", index=False
    )
    features.to_csv(
        OUTPUT_DIR / "selected_features.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    warnings_frame.to_csv(OUTPUT_DIR / "warnings.tsv", sep="\t", index=False)

    summary_metrics = [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "macro_ovr_roc_auc",
        "macro_average_precision",
        "multiclass_log_loss",
        "selected_genes",
    ]
    summary = metrics.groupby("model", sort=False)[summary_metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(
        OUTPUT_DIR / "model_summary.tsv", sep="\t", index=False
    )

    artifact_names = [
        "outer_fold_metrics.tsv",
        "outer_fold_predictions.tsv",
        "best_hyperparameters.tsv",
        "selected_features.tsv.gz",
        "warnings.tsv",
        "model_summary.tsv",
    ]
    manifest = {
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_scheme": "pam50_excluded",
        "models": MODEL_ORDER,
        "development_n": 756,
        "locked_test_n": 189,
        "locked_test_expression_rows_loaded": 0,
        "locked_test_predictions_generated": 0,
        "candidate_protein_coding_genes_before_exclusion": int(len(protein_coding)),
        "candidate_genes_after_exclusion": int(len(candidate_indices)),
        "excluded_pam50_gene_count": 50,
        "outer_folds": 5,
        "inner_folds": 5,
        "random_seed": seed,
        "split_lock_sha256": sha256(SPLIT_LOCK_PATH),
        "pam50_exclusion_config_sha256": sha256(EXCLUSION_CONFIG_PATH),
        "pam50_gene_table_sha256": sha256(PAM50_GENES_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "matrix_manifest_sha256": split_lock["artifacts_sha256"][
            "data/processed/expression/matrix_manifest.json"
        ],
        "output_sha256": {
            name: sha256(OUTPUT_DIR / name) for name in artifact_names
        },
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
