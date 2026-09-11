#!/usr/bin/env python3
"""Verify LinearSVC outputs, calibration scope, and locked-test isolation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "linear_svc"
ASSIGNMENT_PATH = (
    ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(
        (OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8")
    )
    assignments = pd.read_csv(ASSIGNMENT_PATH, sep="\t")
    development = set(
        assignments.loc[
            assignments["holdout_split"].eq("development"), "case_barcode"
        ]
    )
    locked_test = set(
        assignments.loc[
            assignments["holdout_split"].eq("locked_test"), "case_barcode"
        ]
    )
    metrics = pd.read_csv(OUTPUT_DIR / "outer_fold_metrics.tsv", sep="\t")
    predictions = pd.read_csv(OUTPUT_DIR / "outer_fold_predictions.tsv", sep="\t")
    parameters = pd.read_csv(OUTPUT_DIR / "best_hyperparameters.tsv", sep="\t")
    coefficients = pd.read_csv(
        OUTPUT_DIR / "outer_fold_coefficients.tsv.gz", sep="\t"
    )
    checks: list[str] = []

    assert manifest["status"] == "COMPLETE"
    assert manifest["model"] == "LinearSVC"
    assert manifest["kernel"] == "linear"
    assert manifest["rbf_models_fitted"] == 0
    checks.append("linear_svc_only_no_rbf")
    assert manifest["locked_test_expression_rows_loaded"] == 0
    assert manifest["locked_test_predictions_generated"] == 0
    checks.append("locked_test_not_used")

    assert len(metrics) == 5
    assert set(metrics["outer_fold"]) == {1, 2, 3, 4, 5}
    assert metrics["outer_validation_n"].sum() == 756
    for metric in [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "calibrated_macro_f1",
        "decision_macro_ovr_roc_auc",
        "calibrated_macro_ovr_roc_auc",
    ]:
        assert metrics[metric].between(0, 1).all(), metric
    assert np.isfinite(metrics["calibrated_multiclass_log_loss"]).all()
    checks.append("outer_fold_metrics")

    assert len(predictions) == 756
    assert set(predictions["case_barcode"]) == development
    assert set(predictions["case_barcode"]).isdisjoint(locked_test)
    assert not predictions["case_barcode"].duplicated().any()
    probability_columns = [
        column for column in predictions if column.startswith("probability_")
    ]
    assert len(probability_columns) == 4
    np.testing.assert_allclose(
        predictions[probability_columns].sum(axis=1).to_numpy(), 1.0, atol=1e-6
    )
    checks.append("complete_oof_predictions_and_probabilities")

    assert parameters["classifier_class"].eq("LinearSVC").all()
    assert parameters["kernel"].eq("linear").all()
    assert parameters["class_weight"].eq("balanced").all()
    assert parameters["calibration_class"].eq("CalibratedClassifierCV").all()
    assert parameters["calibration_method"].eq("sigmoid").all()
    assert parameters["calibration_cv_folds"].eq(5).all()
    allowed_c = {0.001, 0.01, 0.1, 1.0, 10.0}
    assert all(
        json.loads(value)["classifier__C"] in allowed_c
        for value in parameters["best_parameters_json"]
    )
    checks.append("training_only_calibration_and_parameters")

    assert len(coefficients) == 5 * 4 * 1000
    assert coefficients.groupby(["outer_fold", "pam50_class"]).size().eq(1000).all()
    checks.append("coefficient_axes")
    assert metrics["convergence_warning_count"].sum() == 0
    checks.append("no_convergence_warnings")

    for name, expected in manifest["output_sha256"].items():
        assert sha256(OUTPUT_DIR / name) == expected, name
    assert sha256(ROOT / "config" / "linear_svc_v1.json") == manifest[
        "linear_svc_config_sha256"
    ]
    assert sha256(ROOT / "scripts" / "run_linear_svc.py") == manifest[
        "runner_sha256"
    ]
    checks.append("artifact_sha256")

    report_manifest = json.loads(
        (OUTPUT_DIR / "report_manifest.json").read_text(encoding="utf-8")
    )
    assert report_manifest["status"] == "COMPLETE"
    assert sha256(ROOT / "scripts" / "summarize_linear_svc.py") == report_manifest[
        "summary_script_sha256"
    ]
    for name, expected in report_manifest["artifacts_sha256"].items():
        assert sha256(OUTPUT_DIR / name) == expected, name
    comparison = pd.read_csv(
        OUTPUT_DIR / "comparison_with_logistic.tsv", sep="\t"
    )
    assert set(comparison["model"]) == {
        "dummy_prior",
        "multinomial_logistic_l2",
        "multinomial_logistic_elastic_net",
        "linear_svc",
    }
    confusion = pd.read_csv(OUTPUT_DIR / "oof_confusion_matrix.tsv", sep="\t")
    assert len(confusion) == 16
    assert confusion["count"].sum() == 756
    checks.append("comparison_report_and_confusion_matrix")

    report = {
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "development_patients": 756,
        "locked_test_patients_used": 0,
        "rbf_models_fitted": 0,
    }
    with (OUTPUT_DIR / "verification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
