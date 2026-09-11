#!/usr/bin/env python3
"""Verify locked splits, nested fold integrity, hashes, and pipeline invariants."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from evaluation_framework import load_config
except ModuleNotFoundError:  # pragma: no cover - package import
    from scripts.evaluation_framework import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "evaluation.json"
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_assignments(
    assignments: pd.DataFrame,
    *,
    expected_n: int,
    expected_development: int,
    expected_test: int,
    outer_folds: int,
    inner_folds: int,
) -> list[str]:
    checks: list[str] = []
    assert len(assignments) == expected_n
    assert assignments["patient_group"].nunique() == expected_n
    counts = assignments["holdout_split"].value_counts().to_dict()
    assert counts == {"development": expected_development, "locked_test": expected_test}
    test = assignments.loc[assignments["holdout_split"].eq("locked_test")]
    development = assignments.loc[assignments["holdout_split"].eq("development")]
    assert test["outer_fold"].isna().all()
    assert development["outer_fold"].notna().all()
    assert set(development["outer_fold"].astype(int)) == set(range(1, outer_folds + 1))
    checks.extend(["cohort_size", "patient_uniqueness", "holdout_counts", "test_fold_seal"])

    for outer_fold in range(1, outer_folds + 1):
        column = f"inner_fold_outer_{outer_fold}"
        outer_valid = development["outer_fold"].astype(int).eq(outer_fold)
        assert development.loc[outer_valid, column].isna().all()
        inner_values = development.loc[~outer_valid, column]
        assert inner_values.notna().all()
        assert set(inner_values.astype(int)) == set(range(1, inner_folds + 1))
    checks.append("nested_fold_completeness")

    for label, group in development.groupby("pam50_standard"):
        assert group["outer_fold"].astype(int).nunique() == outer_folds, label
        outer_counts = group["outer_fold"].astype(int).value_counts()
        assert int(outer_counts.max() - outer_counts.min()) <= 1, label
        for outer_fold in range(1, outer_folds + 1):
            inner_values = group.loc[
                group["outer_fold"].astype(int).ne(outer_fold),
                f"inner_fold_outer_{outer_fold}",
            ]
            assert inner_values.astype(int).nunique() == inner_folds, (label, outer_fold)
            inner_counts = inner_values.astype(int).value_counts()
            assert int(inner_counts.max() - inner_counts.min()) <= 1, (label, outer_fold)
    checks.append("class_stratification_coverage")
    return checks


def main() -> None:
    config = load_config(CONFIG_PATH)
    outer_folds = int(config["nested_cv"]["outer_folds"])
    inner_folds = int(config["nested_cv"]["inner_folds"])
    lock_path = SPLIT_DIR / "split_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks: list[str] = []

    primary_path = SPLIT_DIR / "four_class_split_assignments.tsv"
    five_path = SPLIT_DIR / "five_class_split_assignments.tsv"
    primary = pd.read_csv(primary_path, sep="\t")
    five = pd.read_csv(five_path, sep="\t")
    checks.extend(
        f"four_class:{item}"
        for item in check_assignments(
            primary,
            expected_n=945,
            expected_development=756,
            expected_test=189,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
        )
    )
    checks.extend(
        f"five_class:{item}"
        for item in check_assignments(
            five,
            expected_n=981,
            expected_development=785,
            expected_test=196,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
        )
    )

    shared = five.loc[five["case_barcode"].isin(primary["case_barcode"])]
    merged = primary[["case_barcode", "holdout_split"]].merge(
        shared[["case_barcode", "holdout_split"]],
        on="case_barcode",
        suffixes=("_four", "_five"),
        validate="one_to_one",
    )
    assert merged["holdout_split_four"].equals(merged["holdout_split_five"])
    checks.append("primary_holdout_preserved_in_five_class")

    for relative_path, expected_hash in lock["artifacts_sha256"].items():
        assert sha256(ROOT / relative_path) == expected_hash, relative_path
    checks.append("locked_artifact_sha256")
    assert lock["test_set_model_accessed"] is False
    checks.append("locked_test_not_accessed")

    test_result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_evaluation_framework"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if test_result.returncode != 0:
        raise RuntimeError(test_result.stdout + "\n" + test_result.stderr)
    checks.append("transformer_unit_tests")

    report = {
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "four_class": lock["primary_cohort"],
        "five_class": lock["five_class_sensitivity_cohort"],
        "test_set_model_accessed": False,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "verification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
