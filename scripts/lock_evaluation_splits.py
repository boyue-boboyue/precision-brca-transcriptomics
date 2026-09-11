#!/usr/bin/env python3
"""Create and cryptographically lock patient-level evaluation assignments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    from evaluation_framework import load_config
except ModuleNotFoundError:  # pragma: no cover - package import during tests
    from scripts.evaluation_framework import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "evaluation.json"
SAMPLES_PATH = ROOT / "data" / "processed" / "expression" / "samples.tsv"
LABEL_LOCK_PATH = ROOT / "data" / "processed" / "labels" / "pancanatlas_pam50_lock.json"
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_table(rows: pd.DataFrame, group_column: str, label_column: str) -> pd.DataFrame:
    label_counts = rows.groupby(group_column, sort=True)[label_column].nunique(dropna=False)
    conflicting = label_counts[label_counts.ne(1)]
    if not conflicting.empty:
        raise ValueError(f"Patient groups with conflicting labels: {conflicting.index.tolist()[:5]}")
    return (
        rows[[group_column, label_column]]
        .drop_duplicates()
        .sort_values(group_column)
        .reset_index(drop=True)
    )


def stratified_holdout_groups(
    rows: pd.DataFrame,
    *,
    group_column: str,
    label_column: str,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    patients = patient_table(rows, group_column, label_column)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=seed
    )
    development_index, test_index = next(
        splitter.split(patients[group_column], patients[label_column])
    )
    assignment = {group: "development" for group in patients.loc[development_index, group_column]}
    assignment.update({group: "locked_test" for group in patients.loc[test_index, group_column]})
    return assignment


def extend_with_normal_like(
    rows: pd.DataFrame,
    primary_assignment: dict[str, str],
    *,
    group_column: str,
    label_column: str,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    assignment = dict(primary_assignment)
    patients = patient_table(rows, group_column, label_column)
    unassigned = patients.loc[~patients[group_column].isin(assignment)].copy()
    if unassigned[label_column].nunique() != 1:
        raise ValueError("Sensitivity-only patients must form exactly one additional stratum")
    rng = np.random.default_rng(seed + 5000)
    group_ids = unassigned[group_column].sort_values().to_numpy()
    test_count = int(round(len(group_ids) * test_fraction))
    test_groups = set(rng.choice(group_ids, size=test_count, replace=False).tolist())
    for group in group_ids:
        assignment[group] = "locked_test" if group in test_groups else "development"
    return assignment


def attach_nested_folds(
    rows: pd.DataFrame,
    holdout_assignment: dict[str, str],
    *,
    group_column: str,
    label_column: str,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> pd.DataFrame:
    result = rows.copy()
    result["patient_group"] = result[group_column]
    result["holdout_split"] = result[group_column].map(holdout_assignment)
    if result["holdout_split"].isna().any():
        raise ValueError("At least one patient lacks a holdout assignment")

    patients = patient_table(
        result.loc[result["holdout_split"].eq("development")],
        group_column,
        label_column,
    )
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    outer_map: dict[str, int] = {}
    for fold, (_, validation_index) in enumerate(
        outer.split(patients[group_column], patients[label_column]), start=1
    ):
        for group in patients.loc[validation_index, group_column]:
            outer_map[group] = fold
    result["outer_fold"] = result[group_column].map(outer_map).astype("Int64")

    for outer_fold in range(1, outer_folds + 1):
        column = f"inner_fold_outer_{outer_fold}"
        result[column] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        outer_train_groups = {
            group for group, fold in outer_map.items() if fold != outer_fold
        }
        inner_patients = patients.loc[
            patients[group_column].isin(outer_train_groups)
        ].reset_index(drop=True)
        inner = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + 1000 + outer_fold,
        )
        inner_map: dict[str, int] = {}
        for inner_fold, (_, validation_index) in enumerate(
            inner.split(inner_patients[group_column], inner_patients[label_column]), start=1
        ):
            for group in inner_patients.loc[validation_index, group_column]:
                inner_map[group] = inner_fold
        result[column] = result[group_column].map(inner_map).astype("Int64")

    columns = [
        "matrix_row",
        "case_id",
        "case_barcode",
        "sample_id",
        "sample_barcode",
        "patient_group",
        label_column,
        "holdout_split",
        "outer_fold",
    ] + [f"inner_fold_outer_{fold}" for fold in range(1, outer_folds + 1)]
    return result[columns].sort_values(["case_barcode", "sample_barcode"]).reset_index(drop=True)


def class_balance(assignments: pd.DataFrame, cohort: str, label_column: str) -> pd.DataFrame:
    table = (
        assignments.groupby(["holdout_split", label_column], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    table.insert(0, "cohort", cohort)
    totals = table.groupby(["cohort", "holdout_split"])["n"].transform("sum")
    table["percent_within_split"] = 100 * table["n"] / totals
    return table


def write_assignment(frame: pd.DataFrame, internal_path: Path, output_path: Path) -> None:
    content = frame.to_csv(sep="\t", index=False, na_rep="")
    if internal_path.exists() and internal_path.read_text(encoding="utf-8") != content:
        raise RuntimeError(
            f"Refusing to overwrite a locked assignment with different content: {internal_path}"
        )
    internal_path.write_text(content, encoding="utf-8")
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    config = load_config(CONFIG_PATH)
    seed = int(config["random_seed"])
    group_column = config["group_column"]
    label_column = config["label_column"]
    outer_folds = int(config["nested_cv"]["outer_folds"])
    inner_folds = int(config["nested_cv"]["inner_folds"])
    samples = pd.read_csv(SAMPLES_PATH, sep="\t", low_memory=False)
    eligible = samples["analysis_eligible"].astype(bool)

    primary_classes = config["primary_analysis"]["classes"]
    primary_rows = samples.loc[
        eligible & samples[label_column].isin(primary_classes)
    ].copy()
    primary_holdout = stratified_holdout_groups(
        primary_rows,
        group_column=group_column,
        label_column=label_column,
        test_fraction=float(config["primary_analysis"]["test_fraction"]),
        seed=seed,
    )
    primary = attach_nested_folds(
        primary_rows,
        primary_holdout,
        group_column=group_column,
        label_column=label_column,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
    )

    five_classes = config["sensitivity_analysis"]["classes"]
    five_rows = samples.loc[eligible & samples[label_column].isin(five_classes)].copy()
    five_holdout = extend_with_normal_like(
        five_rows,
        primary_holdout,
        group_column=group_column,
        label_column=label_column,
        test_fraction=float(config["sensitivity_analysis"]["test_fraction"]),
        seed=seed,
    )
    five = attach_nested_folds(
        five_rows,
        five_holdout,
        group_column=group_column,
        label_column=label_column,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    primary_path = SPLIT_DIR / "four_class_split_assignments.tsv"
    five_path = SPLIT_DIR / "five_class_split_assignments.tsv"
    primary_output = OUTPUT_DIR / "four_class_split_assignments.tsv"
    five_output = OUTPUT_DIR / "five_class_split_assignments.tsv"
    write_assignment(primary, primary_path, primary_output)
    write_assignment(five, five_path, five_output)

    balance = pd.concat(
        [
            class_balance(primary, "four_class", label_column),
            class_balance(five, "five_class", label_column),
        ],
        ignore_index=True,
    )
    balance.to_csv(OUTPUT_DIR / "split_class_balance.tsv", sep="\t", index=False)

    artifacts = {
        "config/evaluation.json": sha256(CONFIG_PATH),
        "data/processed/expression/samples.tsv": sha256(SAMPLES_PATH),
        "data/processed/expression/matrix_manifest.json": sha256(
            ROOT / "data" / "processed" / "expression" / "matrix_manifest.json"
        ),
        "data/processed/labels/pancanatlas_pam50_locked.tsv": sha256(
            ROOT / "data" / "processed" / "labels" / "pancanatlas_pam50_locked.tsv"
        ),
        "data/processed/labels/pancanatlas_pam50_lock.json": sha256(LABEL_LOCK_PATH),
        "data/processed/splits/four_class_split_assignments.tsv": sha256(primary_path),
        "data/processed/splits/five_class_split_assignments.tsv": sha256(five_path),
        "scripts/evaluation_framework.py": sha256(ROOT / "scripts" / "evaluation_framework.py"),
        "scripts/lock_evaluation_splits.py": sha256(Path(__file__).resolve()),
        "scripts/run_nested_cv.py": sha256(ROOT / "scripts" / "run_nested_cv.py"),
        "scripts/verify_evaluation_framework.py": sha256(
            ROOT / "scripts" / "verify_evaluation_framework.py"
        ),
        "requirements-modeling.txt": sha256(ROOT / "requirements-modeling.txt"),
    }
    lock_path = SPLIT_DIR / "split_lock.json"
    original_lock = (
        json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    )
    lock = {
        "schema_version": "1.0.0",
        "created_at_utc": (
            original_lock["created_at_utc"]
            if original_lock is not None
            else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
        "random_seed": seed,
        "primary_cohort": {
            "n": int(len(primary)),
            "development_n": int(primary["holdout_split"].eq("development").sum()),
            "locked_test_n": int(primary["holdout_split"].eq("locked_test").sum()),
            "unique_patient_groups": int(primary["patient_group"].nunique()),
        },
        "five_class_sensitivity_cohort": {
            "n": int(len(five)),
            "development_n": int(five["holdout_split"].eq("development").sum()),
            "locked_test_n": int(five["holdout_split"].eq("locked_test").sum()),
            "unique_patient_groups": int(five["patient_group"].nunique()),
            "primary_assignments_preserved": True,
        },
        "nested_cv": config["nested_cv"],
        "artifacts_sha256": artifacts,
        "test_set_model_accessed": False,
    }
    if original_lock is not None and original_lock != lock:
        raise RuntimeError(
            "Existing split lock differs from regenerated content; create a new lock version "
            "instead of overwriting it"
        )
    with lock_path.open("w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (OUTPUT_DIR / "split_lock.json").open("w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    report = f"""# Locked evaluation split report

Generated: {lock['created_at_utc']}  
Random seed: `{seed}`  
Analysis unit: patient (`{group_column}`)

## Locked partitions

- Primary four-class cohort: {len(primary)} patients; {lock['primary_cohort']['development_n']} development and {lock['primary_cohort']['locked_test_n']} locked test.
- Five-class sensitivity cohort: {len(five)} patients; {lock['five_class_sensitivity_cohort']['development_n']} development and {lock['five_class_sensitivity_cohort']['locked_test_n']} locked test.
- The four-class assignments are unchanged inside the five-class sensitivity file; Normal-like patients were added with the same 80/20 target.
- No duplicate patient groups are present in the current case-level matrix. The generator nevertheless assigns folds at patient-group level and rejects conflicting labels within a patient.

## Nested cross-validation

- Outer loop: {outer_folds}-fold stratified patient-level CV on the development set.
- Inner loop: {inner_folds}-fold stratified patient-level CV separately precomputed inside every outer-training partition.
- `outer_fold` identifies the outer validation fold. `inner_fold_outer_K` is populated only for patients eligible for training in outer fold K.
- Locked-test rows have no outer or inner assignment and are rejected by the development-only runner.

## Leakage boundary

Every candidate estimator is a single scikit-learn Pipeline in this order: low-expression filter, median imputation, top-variance feature selection, standard scaling, classifier. Each transformer is fitted by GridSearchCV only on the relevant inner-training fold.

No model was fitted and the locked test set was not evaluated while creating this report.
"""
    (OUTPUT_DIR / "evaluation_framework_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(lock, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
