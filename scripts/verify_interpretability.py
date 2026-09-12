#!/usr/bin/env python3
"""Verify interpretability leakage boundaries, outputs, biology, and figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "interpretability_v1.json"
OUTPUT_DIR = ROOT / "outputs" / "interpretability"
BIO_DIR = OUTPUT_DIR / "biological_validation"
ASSIGNMENT_PATH = ROOT / "data/processed/splits/four_class_split_assignments.tsv"
CASE_PATH = ROOT / "data/processed/interpretability/shap_case_selection.tsv"
BACKGROUND_PATH = ROOT / "data/processed/interpretability/shap_background_selection.tsv"
ACCESS_PATH = ROOT / "data/processed/splits/interpretation_test_access_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_manifest(path: Path, root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "COMPLETE", f"Incomplete manifest: {path}")
    for relative, expected in manifest.get("output_sha256", {}).items():
        artifact = root / relative
        require(artifact.exists(), f"Missing manifest artifact: {artifact}")
        require(sha256(artifact) == expected, f"Manifest hash mismatch: {artifact}")
    return manifest


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    amendment_path = ROOT / "config/interpretability_execution_amendment_v1.json"
    amendment = (
        json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment_path.exists()
        else {"hash_corrections": {}}
    )
    for relative, expected in config["inputs_sha256"].items():
        observed = sha256(ROOT / relative)
        if observed != expected:
            correction = amendment.get("hash_corrections", {}).get(relative, {})
            require(
                correction.get("locked_sha256") == expected
                and correction.get("corrected_sha256") == observed,
                f"Locked input hash mismatch without exact amendment: {relative}",
            )
            require(
                amendment.get("statistical_plan_changed") is False,
                "Execution amendment changed the statistical plan",
            )
    for relative, expected in config["locked_artifacts"].items():
        require(sha256(ROOT / relative) == expected, f"Locked artifact mismatch: {relative}")

    assignments = pd.read_csv(ASSIGNMENT_PATH, sep="\t")
    development = assignments.loc[assignments["holdout_split"].eq("development")]
    locked_test = assignments.loc[assignments["holdout_split"].eq("locked_test")]
    background = pd.read_csv(BACKGROUND_PATH, sep="\t")
    cases = pd.read_csv(CASE_PATH, sep="\t")
    require(len(background) == 100, "SHAP background must contain 100 cases")
    require(background["case_barcode"].nunique() == 100, "SHAP background cases are not unique")
    require(background["holdout_split"].eq("development").all(), "Background leakage")
    require(set(background["matrix_row"]).issubset(set(development["matrix_row"])), "Background is not development-only")
    require(len(cases) == 6 and cases["case_barcode"].nunique() == 6, "Unexpected SHAP case selection")
    require(set(cases["matrix_row"]).issubset(set(locked_test["matrix_row"])), "SHAP case is not locked test")
    require(not set(cases["case_barcode"]) & set(background["case_barcode"]), "Case/background overlap")
    require(cases["selection_role"].str.startswith("correct_high_confidence_").sum() == 4, "Need four correct class cases")
    require(cases["selection_role"].eq("misclassified_high_confidence").sum() == 1, "Need one error case")
    require(cases["selection_role"].eq("lowest_probability_margin").sum() == 1, "Need one boundary case")

    access = json.loads(ACCESS_PATH.read_text(encoding="utf-8"))
    require(access.get("status") == "COMPLETE", "Interpretation access incomplete")
    require(access.get("locked_test_expression_rows_loaded") == 6, "Only six locked-test rows may be loaded for SHAP")
    require(access.get("performance_metrics_computed") == 0, "Interpretation reran test performance")
    require(access.get("model_selection_performed") is False, "Interpretation performed model selection")

    main_manifest = verify_manifest(OUTPUT_DIR / "run_manifest.json", OUTPUT_DIR)
    require(main_manifest.get("permutation_locked_test_expression_rows_loaded") == 0, "Permutation touched locked test")
    audit = pd.read_csv(OUTPUT_DIR / "rf_permutation_partition_audit.tsv", sep="\t")
    require(set(audit["outer_fold"]) == set(range(1, 6)), "Permutation folds incomplete")
    require(audit["case_overlap_n"].eq(0).all(), "Permutation train/validation overlap")
    require(audit["locked_test_expression_rows_loaded"].eq(0).all(), "Permutation loaded locked-test rows")
    require(audit["selected_gene_n"].eq(2000).all(), "Permutation feature count changed")
    require(audit["repeats"].eq(config["permutation_importance"]["repeats"]).all(), "Permutation repeat mismatch")

    permutation = pd.read_csv(
        OUTPUT_DIR / "rf_outer_validation_permutation_importance.tsv.gz", sep="\t"
    )
    require(len(permutation) == 5 * 2000, "Unexpected permutation importance row count")
    require(permutation.groupby("outer_fold")["matrix_column"].nunique().eq(2000).all(), "Permutation feature axes invalid")
    require(permutation.groupby("outer_fold")["in_macro_f1_top_20"].sum().eq(20).all(), "Each fold must have exactly 20 top features")

    coefficients = pd.read_csv(
        OUTPUT_DIR / "elastic_net_coefficient_summary.tsv.gz", sep="\t"
    )
    require(coefficients["pam50_class"].nunique() == 4, "Elastic-net classes incomplete")
    require(coefficients.groupby("pam50_class")["matrix_column"].nunique().eq(19962).all(), "Elastic-net background axes incomplete")
    elastic_stability = pd.read_csv(OUTPUT_DIR / "elastic_net_top20_stability.tsv", sep="\t")
    require(elastic_stability["top20_outer_fold_count"].between(1, 5).all(), "Invalid Elastic-net stability")
    rf_stability = pd.read_csv(OUTPUT_DIR / "rf_top20_stability.tsv", sep="\t")
    require(rf_stability["top20_outer_fold_count"].between(1, 5).all(), "Invalid RF stability")

    shap_summary = pd.read_csv(OUTPUT_DIR / "shap_case_summary.tsv", sep="\t")
    shap_values = pd.read_csv(OUTPUT_DIR / "shap_values_selected_cases.tsv.gz", sep="\t")
    require(len(shap_summary) == 6, "SHAP summary case count invalid")
    require(shap_summary["absolute_additivity_residual"].max() <= 1e-5, "SHAP additivity failed")
    require(len(shap_values) == 6 * 2000 * 4, "SHAP output shape invalid")
    require(shap_values["case_barcode"].nunique() == 6, "SHAP cases invalid")
    require(shap_values["explained_class"].nunique() == 4, "SHAP class outputs incomplete")

    bio_manifest = verify_manifest(BIO_DIR / "run_manifest.json", BIO_DIR)
    stable = pd.read_csv(BIO_DIR / "stable_important_genes.tsv", sep="\t")
    minimum = config["biological_validation"]["stable_gene_minimum_outer_folds"]
    require(stable["maximum_top20_fold_count"].ge(minimum).all(), "Unstable gene in target set")
    stable_long = pd.read_csv(
        BIO_DIR / "stable_gene_expression_by_development_sample.tsv.gz", sep="\t"
    )
    require(len(stable_long) == len(stable) * 756, "Stable-gene expression rows incomplete")
    require(stable_long["case_barcode"].nunique() == 756, "Biological validation is not development-only")
    require(set(stable_long["case_barcode"]).issubset(set(development["case_barcode"])), "Biological validation leaked test cases")
    subtype = pd.read_csv(BIO_DIR / "stable_gene_subtype_tests.tsv", sep="\t")
    require(len(subtype) == len(stable), "Subtype test gene count invalid")
    require(subtype["bh_q_value"].between(0, 1).all(), "Subtype q-values invalid")
    receptor = pd.read_csv(BIO_DIR / "stable_gene_receptor_associations.tsv", sep="\t")
    require(len(receptor) == 3 * len(stable), "Receptor test rows incomplete")
    require(set(receptor["receptor"]) == {"ER", "PR", "HER2"}, "Receptor axes incomplete")
    finite_q = receptor["bh_q_value"].dropna()
    require(finite_q.between(0, 1).all(), "Receptor q-values invalid")
    overlap = pd.read_csv(BIO_DIR / "stable_gene_pam50_overlap.tsv", sep="\t")
    pam50 = pd.read_csv(ROOT / "data/processed/labels/pam50_signature_genes_v1.tsv", sep="\t")
    expected_overlap = overlap["gene_name"].isin(set(pam50["current_symbol"]))
    require(np.array_equal(expected_overlap, overlap["is_locked_pam50_gene"].astype(bool)), "PAM50 overlap mismatch")
    classification = pd.read_csv(BIO_DIR / "stable_gene_evidence_classification.tsv", sep="\t")
    allowed = set(config["biological_validation"]["classification_rules"])
    require(set(classification["evidence_category"]).issubset(allowed), "Unexpected evidence category")

    request = json.loads((BIO_DIR / "functional_enrichment_request.json").read_text(encoding="utf-8"))
    require(request["domain_scope"] == "custom", "Enrichment did not use a custom background")
    require(set(request["sources"]) == {"GO:BP", "REAC"}, "Enrichment source mismatch")
    require(set(request["query_genes"]) == set(stable["gene_name"]), "Enrichment query differs from stable genes")
    require(set(request["query_genes"]).issubset(set(request["primary_background_genes"])), "Query not contained in primary background")
    require(request["primary_background_gene_count"] >= request["sensitivity_background_gene_count"], "Background nesting invalid")
    enrichment = pd.read_csv(BIO_DIR / "functional_enrichment_results.tsv", sep="\t")
    require(set(enrichment["analysis"]).issubset({"primary_union_background", "sensitivity_all_five_folds_background"}), "Unexpected enrichment analysis")
    require(enrichment["adjusted_p_value"].dropna().between(0, 0.05).all(), "g:Profiler returned result beyond locked threshold")

    report_manifest = verify_manifest(OUTPUT_DIR / "report_manifest.json", OUTPUT_DIR)
    require((OUTPUT_DIR / "interpretability_report.md").exists(), "Interpretability report missing")
    for relative in report_manifest["output_sha256"]:
        if relative.endswith(".png"):
            with Image.open(OUTPUT_DIR / relative) as image:
                require(image.width >= 600 and image.height >= 350, f"Figure too small: {relative}")

    report = {
        "status": "PASS",
        "checks": {
            "locked_input_hashes": "PASS",
            "shap_background_development_only": "PASS",
            "preselected_test_cases_only": "PASS",
            "permutation_outer_validation_only": "PASS",
            "elastic_net_cross_fold_stability": "PASS",
            "rf_cross_fold_stability": "PASS",
            "shap_probability_additivity": "PASS",
            "stable_gene_expression_development_only": "PASS",
            "pam50_overlap": "PASS",
            "custom_expression_filter_enrichment_background": "PASS",
            "receptor_association_multiple_testing": "PASS",
            "evidence_classification": "PASS",
            "artifact_hashes_and_figures": "PASS",
        },
        "summary": {
            "stable_gene_n": len(stable),
            "shap_case_n": len(shap_summary),
            "shap_background_n": len(background),
            "permutation_fold_n": len(audit),
            "enrichment_term_n": len(enrichment),
            "main_figure_n": report_manifest["main_figure_n"],
            "distribution_figure_n": report_manifest[
                "stable_gene_distribution_figure_n"
            ],
        },
    }
    (OUTPUT_DIR / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
