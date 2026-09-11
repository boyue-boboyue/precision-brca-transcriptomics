#!/usr/bin/env python3
"""Verify unified OOF reporting, PAM50 exclusion and one-time final evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "performance_report"
FINAL_DIR = ROOT / "outputs" / "final_evaluation"
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


def verify_hash_map(base: Path, values: dict[str, str]) -> None:
    for relative_path, expected in values.items():
        path = base / relative_path
        if sha256(path) != expected:
            raise AssertionError(f"Hash mismatch: {path}")


def main() -> None:
    checks: list[str] = []
    signature = pd.read_csv(
        ROOT / "data/processed/labels/pam50_signature_genes_v1.tsv", sep="\t"
    )
    assert len(signature) == signature["matrix_column"].nunique() == 50
    assert set(signature.loc[signature["source_symbol"].ne(signature["current_symbol"]), "source_symbol"]) == {
        "CDCA1", "KNTC2", "ORC6L"
    }
    checks.append("pam50_signature_50_unique_genes_and_aliases")

    exclusion_lock = json.loads(
        (ROOT / "config/pam50_exclusion_v1.json").read_text(encoding="utf-8")
    )
    assert sha256(ROOT / exclusion_lock["locked_gene_table"]) == exclusion_lock[
        "locked_gene_table_sha256"
    ]
    verify_hash_map(ROOT, exclusion_lock["inputs_sha256"])
    checks.append("pam50_exclusion_lock_hashes")

    excluded_manifest = json.loads(
        (ROOT / "outputs/modeling/pam50_excluded/run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert excluded_manifest["status"] == "COMPLETE"
    assert excluded_manifest["locked_test_expression_rows_loaded"] == 0
    assert excluded_manifest["locked_test_predictions_generated"] == 0
    assert excluded_manifest["excluded_pam50_gene_count"] == 50
    verify_hash_map(
        ROOT / "outputs/modeling/pam50_excluded", excluded_manifest["output_sha256"]
    )
    excluded_predictions = pd.read_csv(
        ROOT / "outputs/modeling/pam50_excluded/outer_fold_predictions.tsv", sep="\t"
    )
    assert set(excluded_predictions["model"]) == set(MODEL_ORDER)
    assert excluded_predictions.groupby("model").size().eq(756).all()
    assert not excluded_predictions.duplicated(["model", "case_barcode"]).any()
    selected = pd.read_csv(
        ROOT / "outputs/modeling/pam50_excluded/selected_features.tsv.gz", sep="\t"
    )
    assert not selected["is_locked_pam50_gene"].astype(bool).any()
    assert not set(selected["matrix_column"]) & set(signature["matrix_column"])
    checks.append("pam50_excluded_oof_complete_no_signature_leakage")

    summary = pd.read_csv(REPORT_DIR / "nested_cv_summary.tsv", sep="\t")
    per_class = pd.read_csv(REPORT_DIR / "oof_per_class_metrics.tsv", sep="\t")
    confusions = pd.read_csv(REPORT_DIR / "oof_confusion_matrices.tsv", sep="\t")
    roc_auc = pd.read_csv(REPORT_DIR / "oof_roc_auc.tsv", sep="\t")
    pr_auc = pd.read_csv(REPORT_DIR / "oof_pr_auc.tsv", sep="\t")
    differences = pd.read_csv(
        REPORT_DIR / "pam50_included_excluded_differences.tsv", sep="\t"
    )
    assert len(summary) == 10
    assert len(per_class) == 40
    assert len(confusions) == 160
    assert len(roc_auc) == len(pr_auc) == 40
    assert len(differences) == 20
    assert set(summary["feature_scheme"]) == {"pam50_included", "pam50_excluded"}
    assert set(summary["model"]) == set(MODEL_ORDER)
    assert confusions.groupby(["feature_scheme", "model"])["count"].sum().eq(756).all()
    assert roc_auc["ovr_roc_auc"].between(0, 1).all()
    assert pr_auc["average_precision"].between(0, 1).all()
    assert pr_auc["prevalence"].between(0, 1).all()
    checks.append("unified_oof_tables_complete")

    final_lock = json.loads(FINAL_LOCK_PATH.read_text(encoding="utf-8"))
    assert final_lock["selected_model"] == "random_forest"
    assert final_lock["selection_margin_over_runner_up_macro_f1"] >= final_lock[
        "selection_rule"
    ]["tie_threshold_macro_f1"]
    assert final_lock["tie_break_invoked"] is False
    verify_hash_map(ROOT, final_lock["inputs_sha256"])
    assert sha256(ROOT / final_lock["locked_test"]["runner"]) == final_lock[
        "locked_test"
    ]["runner_sha256"]
    checks.append("preregistered_model_selection_applied")

    access = json.loads(
        (ROOT / "data/processed/splits/final_test_access_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert access["status"] == "COMPLETE"
    assert access["evaluation_attempt"] == 1
    assert access["locked_test_expression_rows_loaded"] == 189
    assert access["locked_test_predictions_generated"] == 189
    assert sha256(ROOT / access["prediction_file"]) == access["prediction_sha256"]

    assignments = pd.read_csv(
        ROOT / "data/processed/splits/four_class_split_assignments.tsv", sep="\t"
    )
    test_cases = set(
        assignments.loc[assignments["holdout_split"].eq("locked_test"), "case_barcode"]
    )
    development_cases = set(
        assignments.loc[assignments["holdout_split"].eq("development"), "case_barcode"]
    )
    predictions = pd.read_csv(FINAL_DIR / "locked_test_predictions.tsv", sep="\t")
    assert len(predictions) == predictions["case_barcode"].nunique() == 189
    assert set(predictions["case_barcode"]) == test_cases
    assert not set(predictions["case_barcode"]) & development_cases
    probability_columns = [
        f"probability_{label.lower().replace('-', '_').replace(' ', '_')}"
        for label in CLASS_ORDER
    ]
    probabilities = predictions[probability_columns].to_numpy(float)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    checks.append("locked_test_exactly_once_and_partition_integrity")

    final_summary = pd.read_csv(FINAL_DIR / "locked_test_summary.tsv", sep="\t").iloc[0]
    assert np.isclose(
        final_summary["macro_f1"],
        f1_score(predictions["observed"], predictions["predicted"], average="macro"),
    )
    assert np.isclose(
        final_summary["balanced_accuracy"],
        balanced_accuracy_score(predictions["observed"], predictions["predicted"]),
    )
    final_confusion = pd.read_csv(
        FINAL_DIR / "locked_test_confusion_matrix.tsv", sep="\t"
    )
    assert final_confusion.drop(columns="observed").to_numpy().sum() == 189
    bootstrap = pd.read_csv(FINAL_DIR / "bootstrap_metrics.tsv.gz", sep="\t")
    ci = pd.read_csv(FINAL_DIR / "bootstrap_confidence_intervals.tsv", sep="\t")
    assert bootstrap["bootstrap_repetition"].nunique() == 1000
    assert np.isfinite(bootstrap["value"]).all()
    assert (ci["ci_lower_95"] <= ci["estimate"]).all()
    assert (ci["estimate"] <= ci["ci_upper_95"]).all()
    assert ci["bootstrap_repetitions_effective"].eq(1000).all()
    checks.append("locked_test_metrics_and_stratified_bootstrap")

    final_manifest = json.loads((FINAL_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["locked_test_evaluation_count"] == 1
    verify_hash_map(FINAL_DIR, final_manifest["output_sha256"])
    report_manifest = json.loads((REPORT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    assert report_manifest["status"] == "COMPLETE"
    assert report_manifest["locked_test_evaluation_count"] == 1
    verify_hash_map(ROOT, report_manifest["inputs_sha256"])
    verify_hash_map(REPORT_DIR, report_manifest["outputs_sha256"])
    assert (REPORT_DIR / "unified_performance_report.md").stat().st_size > 1000
    assert len(list((REPORT_DIR / "figures").glob("*.png"))) == 9
    checks.append("manifests_hashes_report_and_figures")

    report = {
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "development_n_per_model_scheme": 756,
        "locked_test_n": 189,
        "locked_test_evaluation_count": 1,
        "selected_model": "random_forest",
    }
    (REPORT_DIR / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


FINAL_LOCK_PATH = ROOT / "config" / "final_model_lock_v1.json"


if __name__ == "__main__":
    main()
