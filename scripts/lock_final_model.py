#!/usr/bin/env python3
"""Apply the preregistered development-only rule and freeze the final model."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "config" / "final_model_lock_v1.json"
EVALUATION_CONFIG_PATH = ROOT / "config" / "evaluation.json"
FOREST_CONFIG_PATH = ROOT / "config" / "random_forest_v1.json"
EXCLUDED_MANIFEST_PATH = (
    ROOT / "outputs" / "modeling" / "pam50_excluded" / "run_manifest.json"
)
INCLUDED_METRIC_PATHS = [
    ROOT / "outputs" / "modeling" / "logistic_comparison" / "outer_fold_metrics.tsv",
    ROOT / "outputs" / "modeling" / "linear_svc" / "outer_fold_metrics.tsv",
    ROOT / "outputs" / "modeling" / "random_forest" / "outer_fold_metrics.tsv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(
            "Final model lock already exists; refusing to rewrite an immutable lock"
        )
    evaluation = json.loads(EVALUATION_CONFIG_PATH.read_text(encoding="utf-8"))
    excluded_manifest = json.loads(EXCLUDED_MANIFEST_PATH.read_text(encoding="utf-8"))
    if excluded_manifest.get("status") != "COMPLETE":
        raise RuntimeError("PAM50-excluded sensitivity analysis is not complete")
    if excluded_manifest.get("locked_test_predictions_generated") != 0:
        raise RuntimeError("Sensitivity analysis unexpectedly accessed the locked test")

    frames = []
    for path in INCLUDED_METRIC_PATHS:
        frame = pd.read_csv(path, sep="\t")
        frames.append(frame[["model", "outer_fold", "macro_f1", "balanced_accuracy"]])
    metrics = pd.concat(frames, ignore_index=True)
    if metrics.groupby("model")["outer_fold"].nunique().ne(5).any():
        raise RuntimeError("Every candidate model must have five outer folds")
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    best = summary.iloc[0]
    runner_up = summary.iloc[1]
    difference = float(best["macro_f1_mean"] - runner_up["macro_f1_mean"])
    threshold = float(evaluation["selection"]["tie_threshold_macro_f1"])
    if best["model"] != "random_forest":
        raise RuntimeError(f"Unexpected selected model: {best['model']}")
    if difference < threshold:
        raise RuntimeError("Tie-break logic is required but is not implemented for this result")

    forest = json.loads(FOREST_CONFIG_PATH.read_text(encoding="utf-8"))
    selection_table = summary.to_dict(orient="records")
    lock = {
        "schema_version": "1.0.0",
        "locked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_feature_scheme": "pam50_included",
        "selection_partition": "development",
        "selection_rule": {
            "primary": "mean outer-fold macro_f1",
            "co_primary_reported": "mean outer-fold balanced_accuracy",
            "tie_threshold_macro_f1": threshold,
            "tie_break_order": evaluation["selection"]["tie_break_order"],
        },
        "candidate_ranking": selection_table,
        "selected_model": "random_forest",
        "selected_estimator": "RandomForestClassifier",
        "selection_margin_over_runner_up_macro_f1": difference,
        "runner_up": str(runner_up["model"]),
        "tie_break_invoked": False,
        "final_training": {
            "development_n": 756,
            "candidate_gene_scheme": "all protein-coding genes; PAM50 included",
            "pipeline_order": evaluation["features"]["pipeline_order"],
            "hyperparameter_selection": (
                "GridSearchCV on the complete development set using the five locked "
                "outer-fold assignments as internal validation folds"
            ),
            "refit_metric": "macro_f1",
            "candidate_parameter_sets": forest["classifier"][
                "candidate_parameter_sets"
            ],
            "class_weight": "balanced",
            "random_seed": int(forest["random_seed"]),
        },
        "locked_test": {
            "n": 189,
            "allowed_evaluation_count": 1,
            "bootstrap_repetitions": 1000,
            "bootstrap_unit": "case",
            "bootstrap_stratified_by": "observed PAM50 class",
            "confidence_interval": "percentile 95%",
            "runner": "scripts/run_final_locked_test.py",
            "runner_sha256": sha256(ROOT / "scripts" / "run_final_locked_test.py"),
        },
        "sensitivity_note": (
            "PAM50-excluded results are reported descriptively and were not used to "
            "override the prespecified PAM50-included primary ranking."
        ),
        "inputs_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                EVALUATION_CONFIG_PATH,
                FOREST_CONFIG_PATH,
                ROOT / "config" / "logistic_comparison_v1.json",
                ROOT / "config" / "linear_svc_v1.json",
                ROOT / "config" / "pam50_exclusion_v1.json",
                EXCLUDED_MANIFEST_PATH,
                *INCLUDED_METRIC_PATHS,
                ROOT / "data" / "processed" / "splits" / "split_lock.json",
            ]
        },
        "locker_sha256": sha256(Path(__file__).resolve()),
    }
    OUTPUT_PATH.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
