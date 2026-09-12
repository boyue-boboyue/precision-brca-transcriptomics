#!/usr/bin/env python3
"""Lock case/background selections and methods before interpretability fitting."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "interpretability_v1.json"
ASSIGNMENT_PATH = (
    ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv"
)
PREDICTION_PATH = ROOT / "outputs" / "final_evaluation" / "locked_test_predictions.tsv"
SELECTION_DIR = ROOT / "data" / "processed" / "interpretability"
CASE_PATH = SELECTION_DIR / "shap_case_selection.tsv"
BACKGROUND_PATH = SELECTION_DIR / "shap_background_selection.tsv"
CLASS_ORDER = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]
PROBABILITY_COLUMNS = {
    "Luminal A": "probability_luminal_a",
    "Luminal B": "probability_luminal_b",
    "Basal-like": "probability_basal_like",
    "HER2-enriched": "probability_her2_enriched",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_label(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def choose_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    probability_columns = list(PROBABILITY_COLUMNS.values())
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    ordered = np.sort(probabilities, axis=1)
    work = predictions.copy()
    work["confidence"] = probabilities.max(axis=1)
    work["probability_margin"] = ordered[:, -1] - ordered[:, -2]
    records: list[pd.Series] = []
    selected_cases: set[str] = set()

    for subtype in CLASS_ORDER:
        candidates = work.loc[
            work["observed"].eq(subtype) & work["predicted"].eq(subtype)
        ].sort_values(
            ["confidence", "probability_margin", "case_barcode"],
            ascending=[False, False, True],
        )
        if candidates.empty:
            raise RuntimeError(f"No correct locked-test prediction for {subtype}")
        row = candidates.iloc[0].copy()
        row["selection_role"] = f"correct_high_confidence_{safe_label(subtype)}"
        records.append(row)
        selected_cases.add(str(row["case_barcode"]))

    errors = work.loc[
        work["observed"].ne(work["predicted"])
        & ~work["case_barcode"].isin(selected_cases)
    ].sort_values(
        ["confidence", "probability_margin", "case_barcode"],
        ascending=[False, False, True],
    )
    if errors.empty:
        raise RuntimeError("No misclassified locked-test case is available")
    error = errors.iloc[0].copy()
    error["selection_role"] = "misclassified_high_confidence"
    records.append(error)
    selected_cases.add(str(error["case_barcode"]))

    boundary = work.loc[~work["case_barcode"].isin(selected_cases)].sort_values(
        ["probability_margin", "confidence", "case_barcode"],
        ascending=[True, True, True],
    )
    if boundary.empty:
        raise RuntimeError("No distinct boundary case is available")
    boundary_row = boundary.iloc[0].copy()
    boundary_row["selection_role"] = "lowest_probability_margin"
    records.append(boundary_row)

    columns = [
        "selection_role",
        "matrix_row",
        "case_barcode",
        "sample_barcode",
        "observed",
        "predicted",
        "confidence",
        "probability_margin",
        *probability_columns,
    ]
    selected = pd.DataFrame(records)[columns].reset_index(drop=True)
    selected.insert(0, "display_order", np.arange(1, len(selected) + 1))
    return selected


def choose_background(assignments: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].copy()
    counts = development["pam50_standard"].value_counts().reindex(CLASS_ORDER)
    exact = counts * n / int(counts.sum())
    allocation = np.floor(exact).astype(int)
    remainder = n - int(allocation.sum())
    fractional = (exact - allocation).sort_values(ascending=False, kind="stable")
    for subtype in fractional.index[:remainder]:
        allocation.loc[subtype] += 1

    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    for subtype in CLASS_ORDER:
        group = development.loc[development["pam50_standard"].eq(subtype)]
        positions = rng.choice(len(group), size=int(allocation.loc[subtype]), replace=False)
        piece = group.iloc[np.sort(positions)].copy()
        piece["background_stratum_n"] = int(allocation.loc[subtype])
        pieces.append(piece)
    background = pd.concat(pieces, ignore_index=True)
    background = background.sort_values(["pam50_standard", "case_barcode"]).reset_index(
        drop=True
    )
    background.insert(0, "background_order", np.arange(1, len(background) + 1))
    return background[
        [
            "background_order",
            "matrix_row",
            "case_barcode",
            "sample_barcode",
            "patient_group",
            "pam50_standard",
            "holdout_split",
            "outer_fold",
            "background_stratum_n",
        ]
    ]


def main() -> None:
    for path in (CONFIG_PATH, CASE_PATH, BACKGROUND_PATH):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite locked artifact: {path}")
    assignments = pd.read_csv(ASSIGNMENT_PATH, sep="\t")
    predictions = pd.read_csv(PREDICTION_PATH, sep="\t")
    test_assignments = assignments.loc[
        assignments["holdout_split"].eq("locked_test")
    ]
    if len(predictions) != 189 or len(test_assignments) != 189:
        raise RuntimeError("Expected 189 locked-test predictions and assignments")
    if set(predictions["matrix_row"]) != set(test_assignments["matrix_row"]):
        raise RuntimeError("Prediction and assignment test rows do not match")
    final_access = json.loads(
        (ROOT / "data/processed/splits/final_test_access_v1.json").read_text(
            encoding="utf-8"
        )
    )
    if final_access.get("status") != "COMPLETE" or final_access.get(
        "evaluation_attempt"
    ) != 1:
        raise RuntimeError("Final locked-test evaluation record is not complete")

    seed = 20260912
    cases = choose_cases(predictions)
    background = choose_background(assignments, n=100, seed=seed)
    if not background["holdout_split"].eq("development").all():
        raise RuntimeError("SHAP background includes non-development cases")
    if set(background["case_barcode"]) & set(cases["case_barcode"]):
        raise RuntimeError("SHAP background overlaps selected locked-test cases")

    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    cases.to_csv(CASE_PATH, sep="\t", index=False)
    background.to_csv(BACKGROUND_PATH, sep="\t", index=False)

    input_paths = [
        "config/evaluation.json",
        "config/final_model_lock_v1.json",
        "config/random_forest_v1.json",
        "data/processed/expression/matrix_manifest.json",
        "data/processed/expression/genes.tsv",
        "data/processed/expression/protein_coding_gene_indices.npy",
        "data/processed/labels/pam50_signature_genes_v1.tsv",
        "data/processed/splits/four_class_split_assignments.tsv",
        "data/processed/splits/final_test_access_v1.json",
        "outputs/eda/tables/analysis_samples.tsv",
        "outputs/final_evaluation/locked_test_predictions.tsv",
        "outputs/final_evaluation/selected_hyperparameters.json",
        "outputs/modeling/logistic_comparison/outer_fold_coefficients.tsv.gz",
        "outputs/modeling/random_forest/best_hyperparameters.tsv",
        "scripts/run_interpretability.py",
        "scripts/run_biological_validation.py",
        "scripts/fetch_functional_enrichment.py",
        "scripts/generate_interpretability_report.py",
        "scripts/verify_interpretability.py",
        "requirements-interpretability.txt",
    ]
    config: dict[str, Any] = {
        "schema_version": "1.0.0",
        "locked_at_utc": now_utc(),
        "analysis_partition": "development outer validation folds plus preselected locked-test cases for explanation only",
        "random_seed": seed,
        "class_order": CLASS_ORDER,
        "elastic_net": {
            "source": "saved outer-fold standardized coefficients",
            "model": "multinomial_logistic_elastic_net",
            "outer_folds": 5,
            "missing_or_unselected_coefficient": 0.0,
            "class_top_positive": 20,
            "class_top_negative": 20,
            "stability_cutoffs": [20, 50, 100],
            "primary_stability_cutoff": 20,
        },
        "permutation_importance": {
            "model": "random_forest",
            "fit_partition": "each locked outer-training fold",
            "evaluation_partition": "corresponding locked outer-validation fold only",
            "locked_test_rows_loaded": 0,
            "scoring": ["macro_f1", "balanced_accuracy"],
            "primary_scoring": "macro_f1",
            "repeats": 5,
            "outer_folds": 5,
            "forest_n_jobs": 1,
            "permutation_n_jobs": 4,
            "stability_cutoffs": [20, 50, 100],
            "primary_stability_cutoff": 20,
            "missing_or_unselected_importance": 0.0,
        },
        "shap": {
            "version": shap.__version__,
            "explainer": "TreeExplainer",
            "feature_perturbation": "interventional",
            "model_output": "probability",
            "model": "frozen final random_forest refit on all development cases",
            "background_partition": "development only",
            "background_n": 100,
            "background_selection": "class-stratified deterministic sample locked before SHAP computation",
            "case_selection": "saved prediction class, correctness, confidence and top-two probability margin only",
            "selected_test_case_n": 6,
            "purpose": "post-evaluation explanation only; no performance scoring, tuning, or model selection",
            "predicted_class_top_contributions": 20,
        },
        "biological_validation": {
            "stable_gene_rule": "top-20 in at least 3 of 5 outer folds for random-forest permutation importance or for any Elastic-net class",
            "stable_gene_minimum_outer_folds": 3,
            "expression_partition": "development only",
            "subtype_test": "Kruskal-Wallis with Benjamini-Hochberg correction",
            "pam50_overlap": "locked 50-gene PAM50 signature, including recorded current-symbol alias resolution",
            "enrichment_service": "g:Profiler g:GOSt JSON API",
            "enrichment_sources": ["GO:BP", "REAC"],
            "enrichment_domain_scope": "custom",
            "enrichment_background": "union of protein-coding genes passing the training-only low-expression filter in at least one outer fold",
            "enrichment_background_sensitivity": "genes passing the low-expression filter in all five outer folds",
            "multiple_testing": "Benjamini-Hochberg FDR",
            "fdr_threshold": 0.05,
            "receptor_partition": "development cases with Positive or Negative IHC status",
            "receptor_test": "two-sided Mann-Whitney U with per-receptor Benjamini-Hochberg correction and rank-biserial effect size",
            "classification_rules": {
                "known_marker_or_PAM50": "predeclared expected marker or locked PAM50 signature member",
                "clinically_correlated_non_PAM50": "not known/PAM50 and any receptor q<0.05 with absolute rank-biserial correlation >=0.33",
                "potential_candidate_not_novelty_claim": "remaining stable gene with subtype q<0.05; requires independent functional and cohort validation",
                "other_model_associated": "remaining stable model-associated gene",
            },
        },
        "expected_signal_audit": [
            "ESR1",
            "PGR",
            "ERBB2",
            "FOXA1",
            "GATA3",
            "MKI67",
            "KRT5",
            "KRT14",
            "KRT17",
            "EGFR",
        ],
        "interpretation_caveats": [
            "High correlation can distribute importance across genes; the first-ranked gene is not necessarily uniquely important.",
            "All feature attributions are associational and are not causal effects.",
            "Expected breast-cancer markers are audited but never inserted, promoted, or required to appear.",
            "PAM50 labels are themselves expression-derived, so prediction is not independent biological subtype discovery.",
        ],
        "locked_artifacts": {
            "data/processed/interpretability/shap_case_selection.tsv": sha256(CASE_PATH),
            "data/processed/interpretability/shap_background_selection.tsv": sha256(
                BACKGROUND_PATH
            ),
        },
        "inputs_sha256": {relative: sha256(ROOT / relative) for relative in input_paths},
        "locker_sha256": sha256(Path(__file__).resolve()),
    }
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "LOCKED",
                "config": str(CONFIG_PATH.relative_to(ROOT)),
                "shap_cases": len(cases),
                "background": background["pam50_standard"].value_counts().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
