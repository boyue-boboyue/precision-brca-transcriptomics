#!/usr/bin/env python3
"""Verify robustness outputs, split isolation, and artifact hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from run_robustness_analysis import FIVE_CLASSES, load_json, scenario_definitions
except ModuleNotFoundError:  # pragma: no cover
    from scripts.run_robustness_analysis import (
        FIVE_CLASSES,
        load_json,
        scenario_definitions,
    )


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "robustness"
CONFIG_PATH = ROOT / "config" / "robustness_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    checks: list[dict[str, Any]] = []
    config = load_json(CONFIG_PATH)
    expected_scenarios = {
        item["scenario"]: item for item in scenario_definitions(config)
    }
    manifest = load_json(OUTPUT_DIR / "run_manifest.json")
    metrics = pd.read_csv(OUTPUT_DIR / "scenario_fold_metrics.tsv", sep="\t")
    summary = pd.read_csv(OUTPUT_DIR / "scenario_summary.tsv", sep="\t")
    predictions = pd.read_csv(
        OUTPUT_DIR / "scenario_oof_predictions.tsv.gz", sep="\t"
    )
    split_audit = pd.read_csv(OUTPUT_DIR / "split_integrity.tsv", sep="\t")
    per_class = pd.read_csv(OUTPUT_DIR / "scenario_per_class_metrics.tsv", sep="\t")
    confusion = pd.read_csv(OUTPUT_DIR / "scenario_confusion_matrices.tsv", sep="\t")
    label_audit = pd.read_csv(OUTPUT_DIR / "label_source_audit.tsv", sep="\t")

    require(manifest["status"] == "COMPLETE", "run manifest is COMPLETE", checks)
    require(
        manifest["analysis_partition"] == "development_only",
        "analysis is explicitly development-only",
        checks,
    )
    require(
        manifest["locked_test_expression_rows_loaded"] == 0,
        "zero locked-test expression rows loaded",
        checks,
    )
    require(
        manifest["locked_test_predictions_generated"] == 0,
        "zero locked-test predictions generated",
        checks,
    )
    require(
        manifest["locked_test_performance_metrics_generated"] == 0,
        "zero locked-test performance metrics generated",
        checks,
    )
    require(
        set(metrics["scenario"]) == set(expected_scenarios),
        "all configured robustness scenarios are present",
        checks,
    )
    require(len(metrics) == 5 * len(expected_scenarios), "five folds per scenario", checks)
    require(
        (metrics.groupby("scenario")["outer_fold"].nunique() == 5).all(),
        "each scenario has five unique folds",
        checks,
    )
    require(
        set(summary["scenario"]) == set(expected_scenarios),
        "summary contains every scenario exactly once",
        checks,
    )
    require(
        np.isfinite(
            metrics[
                [
                    "macro_f1",
                    "balanced_accuracy",
                    "accuracy",
                    "macro_ovr_roc_auc",
                    "macro_average_precision",
                    "multiclass_log_loss",
                ]
            ].to_numpy()
        ).all(),
        "all reported fold metrics are finite",
        checks,
    )
    require(
        int(split_audit["patient_overlap_count"].max()) == 0,
        "no patient group crosses a train/validation boundary",
        checks,
    )
    require(
        int(split_audit["maximum_samples_per_case"].max()) == 1,
        "current development matrices contain one sample per case",
        checks,
    )

    four_assignment = pd.read_csv(
        ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv",
        sep="\t",
    )
    five_assignment = pd.read_csv(
        ROOT / "data" / "processed" / "splits" / "five_class_split_assignments.tsv",
        sep="\t",
    )
    locked_cases = set(
        four_assignment.loc[
            four_assignment["holdout_split"].eq("locked_test"), "case_barcode"
        ]
    ) | set(
        five_assignment.loc[
            five_assignment["holdout_split"].eq("locked_test"), "case_barcode"
        ]
    )
    require(
        not (set(predictions["case_barcode"]) & locked_cases),
        "OOF prediction table contains no locked-test case",
        checks,
    )
    for scenario_name, definition in expected_scenarios.items():
        subset = predictions.loc[predictions["scenario"].eq(scenario_name)]
        expected_n = 756 if definition["cohort"] == "four_class" else 785
        require(
            len(subset) == expected_n and subset["case_barcode"].nunique() == expected_n,
            f"{scenario_name} has one OOF prediction per development case",
            checks,
        )
        labels = FIVE_CLASSES if definition["cohort"] == "five_class" else FIVE_CLASSES[:-1]
        probability_columns = [
            "probability_" + label.lower().replace("-", "_").replace(" ", "_")
            for label in labels
        ]
        require(
            np.allclose(subset[probability_columns].sum(axis=1), 1.0, atol=1e-6),
            f"{scenario_name} probabilities sum to one",
            checks,
        )

    require(
        len(per_class) == sum(
            5 if item["cohort"] == "five_class" else 4
            for item in expected_scenarios.values()
        ),
        "pooled per-class metrics have the expected class rows",
        checks,
    )
    expected_confusion_rows = sum(
        25 if item["cohort"] == "five_class" else 16
        for item in expected_scenarios.values()
    )
    require(
        len(confusion) == expected_confusion_rows,
        "confusion matrices have all observed/predicted class cells",
        checks,
    )

    primary_metrics = metrics.loc[metrics["scenario"].eq("four_class_tpm_primary")].sort_values(
        "outer_fold"
    )
    original_metrics = pd.read_csv(
        ROOT / "outputs" / "modeling" / "random_forest" / "outer_fold_metrics.tsv",
        sep="\t",
    ).sort_values("outer_fold")
    require(
        np.allclose(
            primary_metrics[["macro_f1", "balanced_accuracy", "accuracy"]],
            original_metrics[["macro_f1", "balanced_accuracy", "accuracy"]],
            rtol=0,
            atol=1e-12,
        ),
        "fixed reference rerun reproduces the selected-model outer metrics",
        checks,
    )
    pam50_only = metrics.loc[metrics["scenario"].eq("four_class_pam50_only")]
    excluded = metrics.loc[metrics["scenario"].eq("four_class_pam50_excluded")]
    require(
        int(pam50_only["candidate_gene_count"].unique()[0]) == 50,
        "PAM50-only analysis starts with exactly 50 genes",
        checks,
    )
    require(
        int(excluded["candidate_gene_count"].unique()[0]) == 19912,
        "PAM50-excluded analysis removes exactly 50 protein-coding genes",
        checks,
    )

    internal = label_audit.loc[
        ~label_audit["source_a"].eq("TCGA_2012_publication_PAM50")
    ]
    historical = label_audit.loc[
        label_audit["source_a"].eq("TCGA_2012_publication_PAM50")
    ].iloc[0]
    require(
        np.allclose(internal["exact_concordance"], 1.0),
        "raw, locked, expression-axis, and split labels are exactly concordant",
        checks,
    )
    require(
        int(historical["overlap_cases"]) == 447
        and int(historical["agreement_cases"]) == 398,
        "historical-versus-PanCancer label audit has the expected 447-case overlap",
        checks,
    )

    for relative, expected_hash in manifest["input_sha256"].items():
        require(
            sha256(ROOT / relative) == expected_hash,
            f"input hash matches: {relative}",
            checks,
        )
    require(
        sha256(ROOT / "scripts" / "run_robustness_analysis.py")
        == manifest["runner_sha256"],
        "runner hash matches manifest",
        checks,
    )
    for relative, expected_hash in manifest["output_sha256"].items():
        require(
            sha256(OUTPUT_DIR / relative) == expected_hash,
            f"output hash matches: {relative}",
            checks,
        )

    report = {
        "status": "PASS",
        "check_count": len(checks),
        "scenario_count": len(expected_scenarios),
        "outer_fit_count": len(metrics),
        "locked_test_expression_rows_loaded": 0,
        "locked_test_predictions_generated": 0,
        "maximum_patient_overlap": int(split_audit["patient_overlap_count"].max()),
        "checks": checks,
    }
    (OUTPUT_DIR / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in report if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
