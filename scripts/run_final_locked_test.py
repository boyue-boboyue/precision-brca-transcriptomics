#!/usr/bin/env python3
"""Fit the frozen final model and evaluate the locked test exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize

try:
    from evaluation_framework import build_pipeline, load_config
except ModuleNotFoundError:  # pragma: no cover
    from scripts.evaluation_framework import build_pipeline, load_config


ROOT = Path(__file__).resolve().parents[1]
FINAL_LOCK_PATH = ROOT / "config" / "final_model_lock_v1.json"
EVALUATION_CONFIG_PATH = ROOT / "config" / "evaluation.json"
FOREST_CONFIG_PATH = ROOT / "config" / "random_forest_v1.json"
SPLIT_LOCK_PATH = ROOT / "data" / "processed" / "splits" / "split_lock.json"
ASSIGNMENT_PATH = (
    ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv"
)
GENE_PATH = ROOT / "data" / "processed" / "expression" / "genes.tsv"
ACCESS_RECORD_PATH = (
    ROOT / "data" / "processed" / "splits" / "final_test_access_v1.json"
)
OUTPUT_DIR = ROOT / "outputs" / "final_evaluation"
CLASS_ORDER = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_label(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ordered_log_loss(truth: np.ndarray, probabilities: np.ndarray) -> float:
    positions = {label: index for index, label in enumerate(CLASS_ORDER)}
    true_positions = np.asarray([positions[str(label)] for label in truth], dtype=int)
    selected = probabilities[np.arange(len(truth)), true_positions]
    return float(-np.log(np.clip(selected, 1e-15, 1.0)).mean())


def probability_order(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    positions = {str(label): index for index, label in enumerate(classes)}
    return np.column_stack([probabilities[:, positions[label]] for label in CLASS_ORDER])


def core_metrics(
    truth: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    binary = label_binarize(truth, classes=CLASS_ORDER)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_ovr_roc_auc": float(
            roc_auc_score(binary, probabilities, average="macro")
        ),
        "macro_average_precision": float(
            average_precision_score(binary, probabilities, average="macro")
        ),
        "multiclass_log_loss": ordered_log_loss(truth, probabilities),
    }


def all_bootstrap_metrics(
    truth: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metric, value in core_metrics(truth, predicted, probabilities).items():
        records.append({"metric": metric, "pam50_class": "macro_or_overall", "value": value})
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=CLASS_ORDER, zero_division=0
    )
    binary = label_binarize(truth, classes=CLASS_ORDER)
    for index, label in enumerate(CLASS_ORDER):
        class_values = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "ovr_roc_auc": float(roc_auc_score(binary[:, index], probabilities[:, index])),
            "average_precision": float(
                average_precision_score(binary[:, index], probabilities[:, index])
            ),
        }
        for metric, value in class_values.items():
            records.append({"metric": metric, "pam50_class": label, "value": value})
    return records


def stratified_bootstrap(
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    class_indices = {
        label: np.flatnonzero(truth == label) for label in CLASS_ORDER
    }
    records: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        sampled = np.concatenate(
            [rng.choice(class_indices[label], len(class_indices[label]), replace=True) for label in CLASS_ORDER]
        )
        for result in all_bootstrap_metrics(
            truth[sampled], predicted[sampled], probabilities[sampled]
        ):
            records.append({"bootstrap_repetition": repetition, **result})
    return pd.DataFrame(records)


def verify_locks() -> tuple[dict[str, Any], dict[str, Any]]:
    final_lock = json.loads(FINAL_LOCK_PATH.read_text(encoding="utf-8"))
    if final_lock.get("selected_model") != "random_forest":
        raise RuntimeError("Final lock does not select random_forest")
    for relative_path, expected in final_lock["inputs_sha256"].items():
        if sha256(ROOT / relative_path) != expected:
            raise RuntimeError(f"Final-model input hash mismatch: {relative_path}")
    split_lock = json.loads(SPLIT_LOCK_PATH.read_text(encoding="utf-8"))
    if split_lock.get("test_set_model_accessed") is not False:
        raise RuntimeError("Original split lock was not an untouched-test certificate")
    for relative_path, expected in split_lock["artifacts_sha256"].items():
        if sha256(ROOT / relative_path) != expected:
            raise RuntimeError(f"Locked split artifact hash mismatch: {relative_path}")
    return final_lock, split_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest-n-jobs", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    truth = np.asarray(CLASS_ORDER * 4, dtype=object)
    predicted = truth.copy()
    probabilities = np.full((len(truth), 4), 0.05)
    for row, label in enumerate(truth):
        probabilities[row, CLASS_ORDER.index(str(label))] = 0.85
    metrics = core_metrics(truth, predicted, probabilities)
    assert metrics["macro_f1"] == 1.0
    boot = stratified_bootstrap(truth, predicted, probabilities, 10, 42)
    assert boot["bootstrap_repetition"].nunique() == 10
    assert np.isfinite(boot["value"]).all()
    print("SELF-TEST PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.forest_n_jobs == 0:
        raise ValueError("--forest-n-jobs cannot be zero")
    if ACCESS_RECORD_PATH.exists():
        raise RuntimeError(
            "A locked-test access record already exists; refusing a second evaluation"
        )
    final_lock, split_lock = verify_locks()
    evaluation = load_config(EVALUATION_CONFIG_PATH)
    forest = load_config(FOREST_CONFIG_PATH)
    assignments = pd.read_csv(ASSIGNMENT_PATH, sep="\t")
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    locked_test = assignments.loc[
        assignments["holdout_split"].eq("locked_test")
    ].reset_index(drop=True)
    if len(development) != 756 or len(locked_test) != 189:
        raise RuntimeError("Unexpected final partition sizes")
    if development["case_barcode"].duplicated().any() or locked_test[
        "case_barcode"
    ].duplicated().any():
        raise RuntimeError("Final evaluation requires one sample per case")
    if set(development["case_barcode"]) & set(locked_test["case_barcode"]):
        raise RuntimeError("Patient leakage between development and locked test")

    matrix = np.load(ROOT / evaluation["features"]["matrix"], mmap_mode="r")
    candidate_indices = np.load(
        ROOT / evaluation["features"]["candidate_gene_indices"]
    ).astype(int)
    development_rows = development["matrix_row"].astype(int).to_numpy()
    X_development = np.asarray(
        matrix[development_rows][:, candidate_indices], dtype=np.float32
    )
    y_development = development[evaluation["label_column"]].astype(str).to_numpy()
    fold_values = development["outer_fold"].astype(int).to_numpy()
    cv = [
        (np.flatnonzero(fold_values != fold), np.flatnonzero(fold_values == fold))
        for fold in sorted(np.unique(fold_values))
    ]
    parameter_grid = [
        {key: [value] for key, value in candidate.items()}
        for candidate in forest["classifier"]["candidate_parameter_sets"]
    ]
    pipeline = build_pipeline(
        "random_forest", evaluation, random_seed=int(forest["random_seed"])
    )
    pipeline.set_params(
        classifier__class_weight="balanced",
        classifier__criterion="gini",
        classifier__bootstrap=True,
        classifier__n_jobs=args.forest_n_jobs,
    )
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
        refit="macro_f1",
        cv=cv,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    print(f"[{now_utc()}] Full-development tuning started", flush=True)
    search.fit(X_development, y_development)
    best = search.best_estimator_
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_results = pd.DataFrame(search.cv_results_)
    cv_columns = [
        column
        for column in cv_results.columns
        if column.startswith("param_")
        or column.startswith("mean_test_")
        or column.startswith("std_test_")
        or column.startswith("rank_test_")
    ]
    cv_results[cv_columns].to_csv(
        OUTPUT_DIR / "full_development_tuning.tsv", sep="\t", index=False
    )
    (OUTPUT_DIR / "selected_hyperparameters.json").write_text(
        json.dumps(
            {
                "best_parameters": search.best_params_,
                "best_internal_macro_f1": float(search.best_score_),
                "development_n": 756,
                "folds": 5,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    low = best.named_steps["low_expression_filter"].get_support(indices=True)
    variance = best.named_steps["variance_selector"].get_support(indices=True)
    selected_columns = candidate_indices[low[variance]]
    genes = pd.read_csv(GENE_PATH, sep="\t").set_index("matrix_column")
    selected_genes = genes.loc[selected_columns].reset_index()
    selected_genes["impurity_importance"] = best.named_steps[
        "classifier"
    ].feature_importances_
    selected_genes.to_csv(
        OUTPUT_DIR / "final_selected_features.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    # The append-only access record is created immediately before materializing any
    # locked-test expression row. Its existence prevents a second model evaluation.
    access_record = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "started_at_utc": now_utc(),
        "evaluation_attempt": 1,
        "model": "random_forest",
        "locked_test_n": 189,
        "final_model_lock_sha256": sha256(FINAL_LOCK_PATH),
        "split_lock_sha256": sha256(SPLIT_LOCK_PATH),
        "selected_hyperparameters_sha256": sha256(
            OUTPUT_DIR / "selected_hyperparameters.json"
        ),
    }
    ACCESS_RECORD_PATH.write_text(
        json.dumps(access_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[{now_utc()}] Locked-test evaluation attempt 1 started", flush=True)
    test_rows = locked_test["matrix_row"].astype(int).to_numpy()
    X_test = np.asarray(matrix[test_rows][:, candidate_indices], dtype=np.float32)
    y_test = locked_test[evaluation["label_column"]].astype(str).to_numpy()
    predicted = best.predict(X_test)
    probabilities = probability_order(
        best.predict_proba(X_test), best.named_steps["classifier"].classes_
    )
    prediction_frame = locked_test[
        ["matrix_row", "case_barcode", "sample_barcode"]
    ].copy()
    prediction_frame["observed"] = y_test
    prediction_frame["predicted"] = predicted
    for index, label in enumerate(CLASS_ORDER):
        prediction_frame[f"probability_{safe_label(label)}"] = probabilities[:, index]
    prediction_frame.to_csv(
        OUTPUT_DIR / "locked_test_predictions.tsv", sep="\t", index=False
    )
    access_record.update(
        {
            "status": "COMPLETE",
            "completed_at_utc": now_utc(),
            "locked_test_expression_rows_loaded": 189,
            "locked_test_predictions_generated": 189,
            "prediction_file": "outputs/final_evaluation/locked_test_predictions.tsv",
            "prediction_sha256": sha256(
                OUTPUT_DIR / "locked_test_predictions.tsv"
            ),
        }
    )
    ACCESS_RECORD_PATH.write_text(
        json.dumps(access_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metric_values = core_metrics(y_test, predicted, probabilities)
    pd.DataFrame(
        [{"model": "random_forest", "partition": "locked_test", **metric_values}]
    ).to_csv(OUTPUT_DIR / "locked_test_summary.tsv", sep="\t", index=False)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=CLASS_ORDER, zero_division=0
    )
    binary = label_binarize(y_test, classes=CLASS_ORDER)
    per_class_records = []
    curve_records: list[dict[str, Any]] = []
    for index, label in enumerate(CLASS_ORDER):
        auc = roc_auc_score(binary[:, index], probabilities[:, index])
        ap = average_precision_score(binary[:, index], probabilities[:, index])
        per_class_records.append(
            {
                "pam50_class": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "ovr_roc_auc": float(auc),
                "average_precision": float(ap),
                "prevalence": float(binary[:, index].mean()),
            }
        )
        fpr, tpr, roc_thresholds = roc_curve(binary[:, index], probabilities[:, index])
        for point, (x, y_value, threshold) in enumerate(
            zip(fpr, tpr, roc_thresholds, strict=True)
        ):
            curve_records.append(
                {
                    "curve": "roc",
                    "pam50_class": label,
                    "point": point,
                    "x": float(x),
                    "y": float(y_value),
                    "threshold": float(threshold),
                }
            )
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
            binary[:, index], probabilities[:, index]
        )
        thresholds = np.append(pr_thresholds, np.nan)
        for point, (x, y_value, threshold) in enumerate(
            zip(pr_recall, pr_precision, thresholds, strict=True)
        ):
            curve_records.append(
                {
                    "curve": "precision_recall",
                    "pam50_class": label,
                    "point": point,
                    "x": float(x),
                    "y": float(y_value),
                    "threshold": float(threshold),
                }
            )
    pd.DataFrame(per_class_records).to_csv(
        OUTPUT_DIR / "locked_test_per_class_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(curve_records).to_csv(
        OUTPUT_DIR / "locked_test_curve_points.tsv", sep="\t", index=False
    )
    matrix_values = confusion_matrix(y_test, predicted, labels=CLASS_ORDER)
    pd.DataFrame(matrix_values, index=CLASS_ORDER, columns=CLASS_ORDER).rename_axis(
        "observed"
    ).reset_index().to_csv(
        OUTPUT_DIR / "locked_test_confusion_matrix.tsv", sep="\t", index=False
    )

    repetitions = int(final_lock["locked_test"]["bootstrap_repetitions"])
    bootstrap = stratified_bootstrap(
        y_test,
        predicted,
        probabilities,
        repetitions,
        int(forest["random_seed"]) + 900000,
    )
    bootstrap.to_csv(
        OUTPUT_DIR / "bootstrap_metrics.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    observed_records = all_bootstrap_metrics(y_test, predicted, probabilities)
    observed_lookup = {
        (record["metric"], record["pam50_class"]): record["value"]
        for record in observed_records
    }
    ci = (
        bootstrap.groupby(["metric", "pam50_class"], sort=False)["value"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "ci_lower_95", 0.975: "ci_upper_95"})
    )
    ci["estimate"] = [
        observed_lookup[(row.metric, row.pam50_class)] for row in ci.itertuples()
    ]
    ci["bootstrap_repetitions_requested"] = repetitions
    ci["bootstrap_repetitions_effective"] = bootstrap[
        "bootstrap_repetition"
    ].nunique()
    ci.to_csv(OUTPUT_DIR / "bootstrap_confidence_intervals.tsv", sep="\t", index=False)

    artifact_names = [
        "full_development_tuning.tsv",
        "selected_hyperparameters.json",
        "final_selected_features.tsv.gz",
        "locked_test_predictions.tsv",
        "locked_test_summary.tsv",
        "locked_test_per_class_metrics.tsv",
        "locked_test_curve_points.tsv",
        "locked_test_confusion_matrix.tsv",
        "bootstrap_metrics.tsv.gz",
        "bootstrap_confidence_intervals.tsv",
    ]
    manifest = {
        "status": "COMPLETE",
        "created_at_utc": now_utc(),
        "model": "random_forest",
        "feature_scheme": "pam50_included",
        "development_n": 756,
        "locked_test_n": 189,
        "locked_test_evaluation_count": 1,
        "bootstrap_repetitions_requested": repetitions,
        "bootstrap_repetitions_effective": int(
            bootstrap["bootstrap_repetition"].nunique()
        ),
        "final_model_lock_sha256": sha256(FINAL_LOCK_PATH),
        "split_lock_sha256": sha256(SPLIT_LOCK_PATH),
        "test_access_record_sha256": sha256(ACCESS_RECORD_PATH),
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
    print(pd.DataFrame([metric_values]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
