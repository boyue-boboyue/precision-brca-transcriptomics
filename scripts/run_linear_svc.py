#!/usr/bin/env python3
"""Run leakage-safe LinearSVC nested CV and training-only probability calibration."""

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
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize

try:
    from evaluation_framework import build_pipeline, inner_splits, load_config
except ModuleNotFoundError:  # pragma: no cover - package import
    from scripts.evaluation_framework import build_pipeline, inner_splits, load_config


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CONFIG_PATH = ROOT / "config" / "evaluation.json"
SVC_CONFIG_PATH = ROOT / "config" / "linear_svc_v1.json"
ASSIGNMENT_PATH = (
    ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv"
)
SPLIT_LOCK_PATH = ROOT / "data" / "processed" / "splits" / "split_lock.json"
GENE_PATH = ROOT / "data" / "processed" / "expression" / "genes.tsv"
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "linear_svc"
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
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def safe_label(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def selected_gene_frame(
    pipeline: Any, protein_coding_indices: np.ndarray, genes: pd.DataFrame
) -> pd.DataFrame:
    low_indices = pipeline.named_steps["low_expression_filter"].get_support(indices=True)
    variance_indices = pipeline.named_steps["variance_selector"].get_support(indices=True)
    matrix_columns = protein_coding_indices[low_indices[variance_indices]]
    return genes.set_index("matrix_column").loc[matrix_columns].reset_index()


def main() -> None:
    args = parse_args()
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero")
    split_lock = verify_split_lock()
    evaluation_config = load_config(EVALUATION_CONFIG_PATH)
    svc_config = load_config(SVC_CONFIG_PATH)
    if svc_config["classifier"]["class"] != "LinearSVC":
        raise RuntimeError("Only LinearSVC is permitted by this runner")
    if svc_config["classifier"]["rbf_allowed"] is not False:
        raise RuntimeError("RBF kernels are prohibited in this comparison")

    assignments = pd.read_csv(ASSIGNMENT_PATH, sep="\t")
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    locked_test = assignments.loc[assignments["holdout_split"].eq("locked_test")]
    if len(development) != 756 or len(locked_test) != 189:
        raise RuntimeError("Unexpected locked partition sizes")
    if locked_test["outer_fold"].notna().any():
        raise RuntimeError("Locked-test rows must not have model-selection folds")
    development_rows = development["matrix_row"].astype(int).to_numpy()
    locked_rows = locked_test["matrix_row"].astype(int).to_numpy()
    if np.intersect1d(development_rows, locked_rows).size:
        raise RuntimeError("Development and locked-test matrix rows overlap")

    matrix = np.load(
        ROOT / evaluation_config["features"]["matrix"], mmap_mode="r"
    )
    protein_coding_indices = np.load(
        ROOT / evaluation_config["features"]["candidate_gene_indices"]
    )
    X = np.asarray(matrix[development_rows][:, protein_coding_indices], dtype=np.float32)
    y = development[evaluation_config["label_column"]].astype(str).to_numpy()
    genes = pd.read_csv(GENE_PATH, sep="\t")
    seed = int(svc_config["random_seed"])
    outer_values = development["outer_fold"].astype(int).to_numpy()
    threshold = float(svc_config["coefficient_nonzero_threshold"])

    metric_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    parameter_records: list[dict[str, Any]] = []
    coefficient_records: list[dict[str, Any]] = []
    warning_records: list[dict[str, Any]] = []

    for outer_fold in range(1, 6):
        print(
            f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] "
            f"LinearSVC outer fold {outer_fold}/5",
            flush=True,
        )
        outer_valid_index = np.flatnonzero(outer_values == outer_fold)
        outer_train_index = np.flatnonzero(outer_values != outer_fold)
        calibration_cv = inner_splits(development, outer_fold)
        pipeline = build_pipeline(
            "linear_svm", evaluation_config, random_seed=seed + outer_fold
        )
        pipeline.set_params(variance_selector__k=1000)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=svc_config["classifier"]["parameter_grid"],
            scoring={
                "macro_f1": "f1_macro",
                "balanced_accuracy": "balanced_accuracy",
            },
            refit=svc_config["inner_selection_metric"],
            cv=calibration_cv,
            n_jobs=args.n_jobs,
            return_train_score=False,
            error_score="raise",
        )
        with warnings.catch_warnings(record=True) as caught_search:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(X[outer_train_index], y[outer_train_index])
        best = search.best_estimator_
        raw_predictions = best.predict(X[outer_valid_index])
        decision_scores = best.decision_function(X[outer_valid_index])
        classifier = best.named_steps["classifier"]
        if classifier.__class__.__name__ != "LinearSVC":
            raise RuntimeError("Unexpected non-linear classifier")

        calibrator = CalibratedClassifierCV(
            estimator=clone(best),
            method=svc_config["probability_calibration"]["method"],
            cv=calibration_cv,
            n_jobs=1,
            ensemble=svc_config["probability_calibration"]["ensemble"],
        )
        with warnings.catch_warnings(record=True) as caught_calibration:
            warnings.simplefilter("always", ConvergenceWarning)
            calibrator.fit(X[outer_train_index], y[outer_train_index])
        calibrated_probabilities = calibrator.predict_proba(X[outer_valid_index])
        calibrated_predictions = calibrator.predict(X[outer_valid_index])
        truth = y[outer_valid_index]

        selected_genes = selected_gene_frame(best, protein_coding_indices, genes)
        coefficients = np.asarray(classifier.coef_)
        if coefficients.shape != (4, 1000):
            raise RuntimeError(f"Unexpected coefficient shape: {coefficients.shape}")
        nonzero = np.abs(coefficients) > threshold
        for class_index, subtype in enumerate(classifier.classes_):
            for gene_index, gene in selected_genes.iterrows():
                value = float(coefficients[class_index, gene_index])
                coefficient_records.append(
                    {
                        "outer_fold": outer_fold,
                        "pam50_class": subtype,
                        "matrix_column": int(gene["matrix_column"]),
                        "gene_id": gene["gene_id"],
                        "gene_id_without_version": gene["gene_id_without_version"],
                        "gene_name": gene["gene_name"],
                        "coefficient": value,
                        "absolute_coefficient": abs(value),
                        "is_nonzero": bool(nonzero[class_index, gene_index]),
                    }
                )

        binarized_truth = label_binarize(truth, classes=classifier.classes_)
        calibrated_class_index = {
            label: index for index, label in enumerate(calibrator.classes_)
        }
        calibrated_in_svc_order = np.column_stack(
            [
                calibrated_probabilities[:, calibrated_class_index[label]]
                for label in classifier.classes_
            ]
        )
        fold_warnings = list(caught_search) + list(caught_calibration)
        convergence_warnings = [
            warning
            for warning in fold_warnings
            if issubclass(warning.category, ConvergenceWarning)
        ]
        for phase, caught in [
            ("inner_search", caught_search),
            ("training_only_calibration", caught_calibration),
        ]:
            for warning in caught:
                if issubclass(warning.category, ConvergenceWarning):
                    warning_records.append(
                        {
                            "outer_fold": outer_fold,
                            "phase": phase,
                            "category": warning.category.__name__,
                            "message": str(warning.message),
                        }
                    )

        metric_records.append(
            {
                "model": "linear_svc",
                "outer_fold": outer_fold,
                "outer_train_n": int(len(outer_train_index)),
                "outer_validation_n": int(len(outer_valid_index)),
                "low_expression_pass_genes": int(
                    best.named_steps["low_expression_filter"].get_support().sum()
                ),
                "selected_genes": 1000,
                "unique_nonzero_genes": int(nonzero.any(axis=0).sum()),
                "macro_f1": float(
                    f1_score(truth, raw_predictions, average="macro")
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(truth, raw_predictions)
                ),
                "accuracy": float(accuracy_score(truth, raw_predictions)),
                "calibrated_macro_f1": float(
                    f1_score(truth, calibrated_predictions, average="macro")
                ),
                "decision_macro_ovr_roc_auc": float(
                    roc_auc_score(binarized_truth, decision_scores, average="macro")
                ),
                "calibrated_macro_ovr_roc_auc": float(
                    roc_auc_score(
                        truth,
                        calibrated_in_svc_order,
                        labels=classifier.classes_,
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                "calibrated_multiclass_log_loss": float(
                    log_loss(
                        truth,
                        calibrated_in_svc_order,
                        labels=classifier.classes_,
                    )
                ),
                "inner_best_macro_f1": float(search.best_score_),
                "convergence_warning_count": len(convergence_warnings),
            }
        )
        parameter_records.append(
            {
                "model": "linear_svc",
                "outer_fold": outer_fold,
                "best_inner_macro_f1": float(search.best_score_),
                "best_parameters_json": json.dumps(search.best_params_, sort_keys=True),
                "class_weight": classifier.class_weight,
                "classifier_class": classifier.__class__.__name__,
                "kernel": "linear",
                "calibration_class": calibrator.__class__.__name__,
                "calibration_method": calibrator.method,
                "calibration_cv_folds": len(calibration_cv),
                "max_n_iter": int(np.max(classifier.n_iter_)),
            }
        )
        svc_class_index = {label: index for index, label in enumerate(classifier.classes_)}
        for position, local_index in enumerate(outer_valid_index):
            sample = development.iloc[int(local_index)]
            record: dict[str, Any] = {
                "model": "linear_svc",
                "outer_fold": outer_fold,
                "matrix_row": int(sample["matrix_row"]),
                "case_barcode": sample["case_barcode"],
                "sample_barcode": sample["sample_barcode"],
                "observed": truth[position],
                "predicted": raw_predictions[position],
                "calibrated_predicted": calibrated_predictions[position],
            }
            for label in CLASS_ORDER:
                record[f"decision_{safe_label(label)}"] = float(
                    decision_scores[position, svc_class_index[label]]
                )
                record[f"probability_{safe_label(label)}"] = float(
                    calibrated_probabilities[position, calibrated_class_index[label]]
                )
            prediction_records.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_records)
    predictions = pd.DataFrame(prediction_records)
    parameters = pd.DataFrame(parameter_records)
    coefficients = pd.DataFrame(coefficient_records)
    warning_frame = pd.DataFrame(warning_records)
    metrics.to_csv(OUTPUT_DIR / "outer_fold_metrics.tsv", sep="\t", index=False)
    predictions.to_csv(
        OUTPUT_DIR / "outer_fold_predictions.tsv", sep="\t", index=False
    )
    parameters.to_csv(
        OUTPUT_DIR / "best_hyperparameters.tsv", sep="\t", index=False
    )
    coefficients.to_csv(
        OUTPUT_DIR / "outer_fold_coefficients.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    warning_frame.to_csv(
        OUTPUT_DIR / "convergence_warnings.tsv", sep="\t", index=False
    )

    summary_columns = [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "calibrated_macro_f1",
        "decision_macro_ovr_roc_auc",
        "calibrated_macro_ovr_roc_auc",
        "calibrated_multiclass_log_loss",
        "unique_nonzero_genes",
    ]
    summary_values: dict[str, Any] = {"model": "linear_svc"}
    for column in summary_columns:
        summary_values[f"{column}_mean"] = float(metrics[column].mean())
        summary_values[f"{column}_std"] = float(metrics[column].std(ddof=1))
    summary = pd.DataFrame([summary_values])
    summary.to_csv(OUTPUT_DIR / "model_summary.tsv", sep="\t", index=False)

    per_class_records: list[dict[str, Any]] = []
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["observed"],
        predictions["predicted"],
        labels=CLASS_ORDER,
        zero_division=0,
    )
    for index, subtype in enumerate(CLASS_ORDER):
        per_class_records.append(
            {
                "model": "linear_svc",
                "pam50_class": subtype,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    pd.DataFrame(per_class_records).to_csv(
        OUTPUT_DIR / "oof_per_class_metrics.tsv", sep="\t", index=False
    )

    manifest = {
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "LinearSVC",
        "kernel": "linear",
        "rbf_models_fitted": 0,
        "class_weight": "balanced",
        "development_n": 756,
        "locked_test_n": 189,
        "locked_test_expression_rows_loaded": 0,
        "locked_test_predictions_generated": 0,
        "probability_calibration": "CalibratedClassifierCV sigmoid within outer-training data",
        "outer_folds": 5,
        "inner_folds": 5,
        "random_seed": seed,
        "split_lock_sha256": sha256(SPLIT_LOCK_PATH),
        "evaluation_config_sha256": sha256(EVALUATION_CONFIG_PATH),
        "linear_svc_config_sha256": sha256(SVC_CONFIG_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "matrix_manifest_sha256": split_lock["artifacts_sha256"][
            "data/processed/expression/matrix_manifest.json"
        ],
    }
    artifact_names = [
        "outer_fold_metrics.tsv",
        "outer_fold_predictions.tsv",
        "best_hyperparameters.tsv",
        "outer_fold_coefficients.tsv.gz",
        "convergence_warnings.tsv",
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
