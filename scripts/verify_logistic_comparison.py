#!/usr/bin/env python3
"""Verify the completed development-only logistic comparison."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "logistic_comparison"
ASSIGNMENT_PATH = (
    ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv"
)
EXPECTED_MODELS = {
    "dummy_prior",
    "multinomial_logistic_l2",
    "multinomial_logistic_elastic_net",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    assert manifest["locked_test_expression_rows_loaded"] == 0
    assert manifest["locked_test_predictions_generated"] == 0
    checks.append("locked_test_not_used")
    assert set(metrics["model"]) == EXPECTED_MODELS
    assert len(metrics) == 15
    assert metrics.groupby("model")["outer_fold"].nunique().eq(5).all()
    checks.append("five_outer_folds_per_model")
    for metric in ["macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_roc_auc"]:
        assert metrics[metric].between(0, 1).all(), metric
    assert np.isfinite(metrics["multiclass_log_loss"]).all()
    checks.append("metric_ranges")

    assert len(predictions) == 756 * 3
    assert set(predictions["case_barcode"]).issubset(development)
    assert set(predictions["case_barcode"]).isdisjoint(locked_test)
    assert predictions.groupby("model")["case_barcode"].nunique().eq(756).all()
    assert not predictions.duplicated(["model", "case_barcode"]).any()
    probability_columns = [
        column for column in predictions if column.startswith("probability_")
    ]
    np.testing.assert_allclose(
        predictions[probability_columns].sum(axis=1).to_numpy(), 1.0, atol=1e-6
    )
    checks.append("development_predictions_complete")

    logistic_parameters = parameters.loc[
        parameters["model"].ne("dummy_prior")
    ]
    assert logistic_parameters["class_weight"].eq("balanced").all()
    l2 = parameters.loc[parameters["model"].eq("multinomial_logistic_l2")]
    elastic = parameters.loc[
        parameters["model"].eq("multinomial_logistic_elastic_net")
    ]
    assert all(json.loads(value)["classifier__l1_ratio"] == 0.0 for value in l2["best_parameters_json"])
    assert all(
        0 < json.loads(value)["classifier__l1_ratio"] < 1
        for value in elastic["best_parameters_json"]
    )
    checks.append("regularization_and_class_weight")

    assert set(coefficients["model"]) == {
        "multinomial_logistic_l2",
        "multinomial_logistic_elastic_net",
    }
    assert len(coefficients) == 2 * 5 * 4 * 1000
    assert coefficients.groupby(["model", "outer_fold", "pam50_class"]).size().eq(1000).all()
    checks.append("coefficient_axes")

    for name, expected in manifest["output_sha256"].items():
        assert sha256(OUTPUT_DIR / name) == expected, name
    assert sha256(ROOT / "config" / "logistic_comparison_v1.json") == manifest[
        "comparison_config_sha256"
    ]
    assert sha256(ROOT / "scripts" / "run_logistic_comparison.py") == manifest[
        "runner_sha256"
    ]
    checks.append("artifact_sha256")

    report_manifest = json.loads(
        (OUTPUT_DIR / "report_manifest.json").read_text(encoding="utf-8")
    )
    assert report_manifest["status"] == "COMPLETE"
    assert sha256(ROOT / "scripts" / "summarize_logistic_comparison.py") == report_manifest[
        "summary_script_sha256"
    ]
    for name, expected in report_manifest["artifacts_sha256"].items():
        assert sha256(OUTPUT_DIR / name) == expected, name
    per_class = pd.read_csv(OUTPUT_DIR / "oof_per_class_metrics.tsv", sep="\t")
    assert len(per_class) == 3 * 4
    assert set(per_class["model"]) == EXPECTED_MODELS
    assert per_class.groupby("model")["support"].sum().eq(756).all()
    checks.append("oof_class_metrics_and_report")

    report = {
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "development_patients": 756,
        "locked_test_patients_used": 0,
        "models": sorted(EXPECTED_MODELS),
    }
    with (OUTPUT_DIR / "verification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
