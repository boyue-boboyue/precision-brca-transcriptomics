#!/usr/bin/env python3
"""Verify RandomForest nested-CV outputs and locked-test isolation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "random_forest"
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
    importances = pd.read_csv(
        OUTPUT_DIR / "outer_fold_impurity_importance.tsv.gz", sep="\t"
    )
    config = json.loads(
        (ROOT / "config" / "random_forest_v1.json").read_text(encoding="utf-8")
    )
    checks: list[str] = []

    assert manifest["status"] == "COMPLETE"
    assert manifest["model"] == "RandomForestClassifier"
    assert manifest["class_weight"] == "balanced"
    assert manifest["locked_test_expression_rows_loaded"] == 0
    assert manifest["locked_test_predictions_generated"] == 0
    checks.extend(["random_forest_balanced", "locked_test_not_used"])

    assert len(metrics) == 5
    assert set(metrics["outer_fold"]) == {1, 2, 3, 4, 5}
    assert metrics["outer_validation_n"].sum() == 756
    for metric in ["macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_roc_auc"]:
        assert metrics[metric].between(0, 1).all(), metric
    assert np.isfinite(metrics["multiclass_log_loss"]).all()
    assert set(metrics["selected_genes"]).issubset({500, 1000, 2000})
    checks.append("outer_fold_metrics_and_feature_selection")

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

    assert parameters["classifier_class"].eq("RandomForestClassifier").all()
    assert parameters["class_weight"].eq("balanced").all()
    allowed_candidates = {
        json.dumps(candidate, sort_keys=True)
        for candidate in config["classifier"]["candidate_parameter_sets"]
    }
    for value in parameters["best_parameters_json"]:
        assert json.dumps(json.loads(value), sort_keys=True) in allowed_candidates
    checks.append("locked_parameter_candidates")

    expected_importance_rows = int(metrics["selected_genes"].sum())
    assert len(importances) == expected_importance_rows
    assert importances.groupby("outer_fold")["impurity_importance"].sum().sub(1).abs().lt(1e-8).all()
    checks.append("importance_axes")
    assert metrics["warning_count"].sum() == 0
    checks.append("no_training_warnings")

    for name, expected in manifest["output_sha256"].items():
        assert sha256(OUTPUT_DIR / name) == expected, name
    assert sha256(ROOT / "config" / "random_forest_v1.json") == manifest[
        "random_forest_config_sha256"
    ]
    assert sha256(ROOT / "scripts" / "run_random_forest.py") == manifest[
        "runner_sha256"
    ]
    checks.append("artifact_sha256")

    report_manifest = json.loads(
        (OUTPUT_DIR / "report_manifest.json").read_text(encoding="utf-8")
    )
    assert report_manifest["status"] == "COMPLETE"
    assert sha256(ROOT / "scripts" / "summarize_random_forest.py") == (
        report_manifest["summary_script_sha256"]
    )
    for name, expected in report_manifest["artifacts_sha256"].items():
        assert sha256(OUTPUT_DIR / name) == expected, name
    checks.append("report_artifact_sha256")

    comparison = pd.read_csv(
        OUTPUT_DIR / "comparison_all_models.tsv", sep="\t"
    )
    assert set(comparison["model"]) == {
        "dummy_prior",
        "multinomial_logistic_l2",
        "multinomial_logistic_elastic_net",
        "linear_svc",
        "random_forest",
    }
    assert len(comparison) == 5
    assert comparison.loc[
        comparison["model"].eq("random_forest"), "selected_genes_mean"
    ].iat[0] == 2000
    checks.append("all_model_comparison")

    paired = pd.read_csv(
        OUTPUT_DIR / "paired_comparison_with_linear_models.tsv", sep="\t"
    )
    assert len(paired) == 45
    assert set(paired["outer_fold"]) == {1, 2, 3, 4, 5}
    assert set(paired["metric"]) == {
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
    }
    assert paired["comparison"].nunique() == 3
    checks.append("paired_same_fold_comparisons")

    confusion = pd.read_csv(OUTPUT_DIR / "oof_confusion_matrix.tsv", sep="\t")
    assert len(confusion) == 16
    assert confusion["count"].sum() == 756
    assert set(confusion["observed"]) == {
        "Luminal A",
        "Luminal B",
        "Basal-like",
        "HER2-enriched",
    }
    assert set(confusion["predicted"]) == set(confusion["observed"])
    checks.append("oof_confusion_matrix")

    report = {
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "development_patients": 756,
        "locked_test_patients_used": 0,
        "class_weight": "balanced",
    }
    with (OUTPUT_DIR / "verification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
