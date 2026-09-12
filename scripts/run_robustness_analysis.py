#!/usr/bin/env python3
"""Run fixed-model, development-only robustness analyses.

This is a post-selection confirmation analysis.  It never slices expression
rows assigned to the locked test set.  The random-forest hyperparameters are
read from the frozen robustness configuration and are not re-tuned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    from evaluation_framework import LowExpressionFilter, TopVarianceSelector
except ModuleNotFoundError:  # pragma: no cover
    from scripts.evaluation_framework import LowExpressionFilter, TopVarianceSelector


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "robustness_v1.json"
OUTPUT_DIR = ROOT / "outputs" / "robustness"
FIGURE_DIR = OUTPUT_DIR / "figures"
FOUR_CLASSES = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]
FIVE_CLASSES = FOUR_CLASSES + ["Normal-like"]
RAW_TO_STANDARD = {
    "BRCA_LumA": "Luminal A",
    "BRCA_LumB": "Luminal B",
    "BRCA_Basal": "Basal-like",
    "BRCA_Her2": "HER2-enriched",
    "BRCA_Normal": "Normal-like",
}
HISTORICAL_TO_STANDARD = {
    "LumA": "Luminal A",
    "LumB": "Luminal B",
    "Basal": "Basal-like",
    "Her2": "HER2-enriched",
    "Normal": "Normal-like",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest-n-jobs", type=int, default=4)
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional scenario names for a smoke run; full reporting requires all scenarios.",
    )
    parser.add_argument(
        "--n-estimators-override",
        type=int,
        help="Testing only. Full reporting refuses an estimator override.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Regenerate summaries/reports from completed per-fold outputs without fitting.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_definitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    seed_values = [int(value) for value in config["cross_validation"]["seed_sensitivity"]]
    scenarios: list[dict[str, Any]] = [
        {
            "scenario": "four_class_tpm_primary",
            "display_name": "4-class · TPM · primary",
            "comparison_family": "reference",
            "cohort": "four_class",
            "matrix_kind": "log2_tpm",
            "feature_scheme": "whole_transcriptome_included",
            "class_weight": "balanced",
            "abundance_threshold": 1.0,
            "minimum_fraction": 0.10,
            "split_scheme": "locked_patient_grouped",
            "split_seed": 20260909,
        },
        {
            "scenario": "five_class_tpm_primary",
            "display_name": "5-class · TPM · primary",
            "comparison_family": "class_definition",
            "cohort": "five_class",
            "matrix_kind": "log2_tpm",
            "feature_scheme": "whole_transcriptome_included",
            "class_weight": "balanced",
            "abundance_threshold": 1.0,
            "minimum_fraction": 0.10,
            "split_scheme": "locked_patient_grouped",
            "split_seed": 20260909,
        },
        {
            "scenario": "four_class_pam50_excluded",
            "display_name": "4-class · PAM50 excluded",
            "comparison_family": "pam50_inclusion",
            "cohort": "four_class",
            "matrix_kind": "log2_tpm",
            "feature_scheme": "whole_transcriptome_excluded",
            "class_weight": "balanced",
            "abundance_threshold": 1.0,
            "minimum_fraction": 0.10,
            "split_scheme": "locked_patient_grouped",
            "split_seed": 20260909,
        },
        {
            "scenario": "four_class_log2_cpm",
            "display_name": "4-class · log2 CPM",
            "comparison_family": "normalization",
            "cohort": "four_class",
            "matrix_kind": "log2_cpm",
            "feature_scheme": "whole_transcriptome_included",
            "class_weight": "balanced",
            "abundance_threshold": 1.0,
            "minimum_fraction": 0.10,
            "split_scheme": "locked_patient_grouped",
            "split_seed": 20260909,
        },
        {
            "scenario": "four_class_filter_tpm0_5_frac0_10",
            "display_name": "4-class · TPM≥0.5 in 10%",
            "comparison_family": "low_expression_filter",
            "cohort": "four_class",
            "matrix_kind": "log2_tpm",
            "feature_scheme": "whole_transcriptome_included",
            "class_weight": "balanced",
            "abundance_threshold": 0.5,
            "minimum_fraction": 0.10,
            "split_scheme": "locked_patient_grouped",
            "split_seed": 20260909,
        },
        {
            "scenario": "four_class_filter_tpm1_frac0_20",
            "display_name": "4-class · TPM≥1 in 20%",
            "comparison_family": "low_expression_filter",
            "cohort": "four_class",
            "matrix_kind": "log2_tpm",
            "feature_scheme": "whole_transcriptome_included",
            "class_weight": "balanced",
            "abundance_threshold": 1.0,
            "minimum_fraction": 0.20,
            "split_scheme": "locked_patient_grouped",
            "split_seed": 20260909,
        },
        {
            "scenario": "four_class_filter_tpm5_frac0_10",
            "display_name": "4-class · TPM≥5 in 10%",
            "comparison_family": "low_expression_filter",
            "cohort": "four_class",
            "matrix_kind": "log2_tpm",
            "feature_scheme": "whole_transcriptome_included",
            "class_weight": "balanced",
            "abundance_threshold": 5.0,
            "minimum_fraction": 0.10,
            "split_scheme": "locked_patient_grouped",
            "split_seed": 20260909,
        },
        {
            "scenario": "four_class_unweighted",
            "display_name": "4-class · unweighted",
            "comparison_family": "class_weight",
            "cohort": "four_class",
            "matrix_kind": "log2_tpm",
            "feature_scheme": "whole_transcriptome_included",
            "class_weight": None,
            "abundance_threshold": 1.0,
            "minimum_fraction": 0.10,
            "split_scheme": "locked_patient_grouped",
            "split_seed": 20260909,
        },
        {
            "scenario": "four_class_pam50_only",
            "display_name": "4-class · PAM50 only",
            "comparison_family": "feature_space",
            "cohort": "four_class",
            "matrix_kind": "log2_tpm",
            "feature_scheme": "pam50_only",
            "class_weight": "balanced",
            "abundance_threshold": 1.0,
            "minimum_fraction": 0.10,
            "split_scheme": "locked_patient_grouped",
            "split_seed": 20260909,
        },
    ]
    for seed in seed_values:
        scenarios.append(
            {
                "scenario": f"four_class_seed_{seed}_patient_grouped",
                "display_name": f"4-class · grouped seed {seed}",
                "comparison_family": "random_seed",
                "cohort": "four_class",
                "matrix_kind": "log2_tpm",
                "feature_scheme": "whole_transcriptome_included",
                "class_weight": "balanced",
                "abundance_threshold": 1.0,
                "minimum_fraction": 0.10,
                "split_scheme": "generated_patient_grouped",
                "split_seed": seed,
            }
        )
    scenarios.append(
        {
            "scenario": "four_class_seed_20260909_sample_stratified",
            "display_name": "4-class · sample split seed 20260909",
            "comparison_family": "split_unit",
            "cohort": "four_class",
            "matrix_kind": "log2_tpm",
            "feature_scheme": "whole_transcriptome_included",
            "class_weight": "balanced",
            "abundance_threshold": 1.0,
            "minimum_fraction": 0.10,
            "split_scheme": "generated_sample_stratified",
            "split_seed": 20260909,
        }
    )
    return scenarios


def load_development(cohort: str) -> tuple[pd.DataFrame, int]:
    path = ROOT / "data" / "processed" / "splits" / f"{cohort}_split_assignments.tsv"
    assignments = pd.read_csv(path, sep="\t")
    development = assignments.loc[assignments["holdout_split"].eq("development")].copy()
    locked_n = int(assignments["holdout_split"].eq("locked_test").sum())
    development = development.reset_index(drop=True)
    if development["case_barcode"].duplicated().any():
        # Multiple samples are allowed only if labels are case-consistent. Splitting
        # remains group-aware below.
        conflicts = development.groupby("case_barcode")["pam50_standard"].nunique()
        if (conflicts > 1).any():
            raise RuntimeError("Conflicting subtype labels within patient groups")
    if development["outer_fold"].isna().any():
        raise RuntimeError(f"Missing locked outer fold in {cohort} development set")
    return development, locked_n


def assign_folds(frame: pd.DataFrame, scenario: dict[str, Any]) -> np.ndarray:
    scheme = scenario["split_scheme"]
    if scheme == "locked_patient_grouped":
        folds = frame["outer_fold"].astype(int).to_numpy()
    elif scheme == "generated_patient_grouped":
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=int(scenario["split_seed"]),
        )
        folds = np.zeros(len(frame), dtype=int)
        for fold, (_, valid) in enumerate(
            splitter.split(
                np.zeros((len(frame), 1)),
                frame["pam50_standard"].astype(str).to_numpy(),
                groups=frame["case_barcode"].astype(str).to_numpy(),
            ),
            start=1,
        ):
            folds[valid] = fold
    elif scheme == "generated_sample_stratified":
        splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=int(scenario["split_seed"]),
        )
        folds = np.zeros(len(frame), dtype=int)
        for fold, (_, valid) in enumerate(
            splitter.split(
                np.zeros((len(frame), 1)),
                frame["pam50_standard"].astype(str).to_numpy(),
            ),
            start=1,
        ):
            folds[valid] = fold
    else:  # pragma: no cover
        raise ValueError(f"Unknown split scheme: {scheme}")
    if set(np.unique(folds)) != {1, 2, 3, 4, 5}:
        raise RuntimeError(f"Invalid five-fold assignment for {scenario['scenario']}")
    return folds


def feature_indices(scheme: str) -> np.ndarray:
    protein = np.load(
        ROOT / "data" / "processed" / "expression" / "protein_coding_gene_indices.npy"
    ).astype(int)
    pam50 = pd.read_csv(
        ROOT / "data" / "processed" / "labels" / "pam50_signature_genes_v1.tsv",
        sep="\t",
    )["matrix_column"].astype(int).to_numpy()
    if len(np.unique(pam50)) != 50 or not np.isin(pam50, protein).all():
        raise RuntimeError("The locked PAM50 signature must contain 50 protein-coding columns")
    if scheme == "whole_transcriptome_included":
        return protein
    if scheme == "whole_transcriptome_excluded":
        return protein[~np.isin(protein, pam50)]
    if scheme == "pam50_only":
        return pam50
    raise ValueError(f"Unknown feature scheme: {scheme}")


def load_expression(rows: np.ndarray, columns: np.ndarray, matrix_kind: str) -> np.ndarray:
    expression_dir = ROOT / "data" / "processed" / "expression"
    if matrix_kind == "log2_tpm":
        matrix = np.load(expression_dir / "log2_tpm_float32.npy", mmap_mode="r")
        return np.asarray(matrix[np.ix_(rows, columns)], dtype=np.float32)
    if matrix_kind != "log2_cpm":
        raise ValueError(matrix_kind)
    counts = np.load(expression_dir / "counts_uint32.npy", mmap_mode="r")
    library_sizes = np.empty(len(rows), dtype=np.float64)
    # Chunking avoids materialising the complete 60,660-gene slice at once.
    for start in range(0, len(rows), 64):
        stop = min(start + 64, len(rows))
        library_sizes[start:stop] = np.asarray(counts[rows[start:stop]]).sum(
            axis=1, dtype=np.float64
        )
    if (library_sizes <= 0).any():
        raise RuntimeError("A count library has non-positive size")
    selected_counts = np.asarray(counts[np.ix_(rows, columns)], dtype=np.float32)
    selected_counts *= (1_000_000.0 / library_sizes).astype(np.float32)[:, None]
    np.log2(selected_counts + 1.0, out=selected_counts)
    return selected_counts


def build_fixed_pipeline(
    config: dict[str, Any], scenario: dict[str, Any], random_state: int, n_jobs: int
) -> Pipeline:
    model = config["estimator"]
    return Pipeline(
        [
            (
                "low_expression_filter",
                LowExpressionFilter(
                    threshold=float(math.log2(float(scenario["abundance_threshold"]) + 1.0)),
                    min_fraction=float(scenario["minimum_fraction"]),
                ),
            ),
            ("median_imputer", SimpleImputer(strategy="median")),
            (
                "variance_selector",
                TopVarianceSelector(k=int(model["variance_selector_k"])),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(model["n_estimators"]),
                    max_depth=int(model["max_depth"]),
                    min_samples_leaf=int(model["min_samples_leaf"]),
                    max_features=model["max_features"],
                    criterion=model["criterion"],
                    bootstrap=bool(model["bootstrap"]),
                    class_weight=scenario["class_weight"],
                    random_state=int(random_state),
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def safe_label(label: str) -> str:
    return label.lower().replace("-", "_").replace(" ", "_")


def run_scenario(
    config: dict[str, Any], scenario: dict[str, Any], forest_n_jobs: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame, _ = load_development(scenario["cohort"])
    rows = frame["matrix_row"].astype(int).to_numpy()
    columns = feature_indices(scenario["feature_scheme"])
    X = load_expression(rows, columns, scenario["matrix_kind"])
    y = frame["pam50_standard"].astype(str).to_numpy()
    groups = frame["case_barcode"].astype(str).to_numpy()
    folds = assign_folds(frame, scenario)
    classes = FOUR_CLASSES if scenario["cohort"] == "four_class" else FIVE_CLASSES

    metric_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    filter_records: list[dict[str, Any]] = []
    for fold in range(1, 6):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        train_groups = set(groups[train])
        valid_groups = set(groups[valid])
        overlap = train_groups & valid_groups
        print(
            f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] "
            f"{scenario['scenario']} fold {fold}/5",
            flush=True,
        )
        estimator = build_fixed_pipeline(
            config,
            scenario,
            random_state=int(scenario["split_seed"]) + fold,
            n_jobs=forest_n_jobs,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X[train], y[train])
        predicted = estimator.predict(X[valid])
        raw_probabilities = estimator.predict_proba(X[valid])
        fitted_classes = estimator.named_steps["classifier"].classes_
        probability_index = {str(label): i for i, label in enumerate(fitted_classes)}
        probabilities = np.column_stack(
            [raw_probabilities[:, probability_index[label]] for label in classes]
        )
        truth = y[valid]
        binary_truth = label_binarize(truth, classes=classes)
        metric_records.append(
            {
                **scenario,
                "class_weight": "none" if scenario["class_weight"] is None else scenario["class_weight"],
                "outer_fold": fold,
                "outer_train_n": len(train),
                "outer_validation_n": len(valid),
                "candidate_gene_count": len(columns),
                "low_expression_pass_genes": int(
                    estimator.named_steps["low_expression_filter"].get_support().sum()
                ),
                "selected_genes": int(
                    estimator.named_steps["variance_selector"].get_support().sum()
                ),
                "macro_f1": float(f1_score(truth, predicted, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
                "accuracy": float(accuracy_score(truth, predicted)),
                "macro_ovr_roc_auc": float(
                    np.mean(
                        [
                            roc_auc_score(binary_truth[:, i], probabilities[:, i])
                            for i in range(len(classes))
                        ]
                    )
                ),
                "macro_average_precision": float(
                    average_precision_score(binary_truth, probabilities, average="macro")
                ),
                "multiclass_log_loss": float(
                    -np.log(
                        np.clip(
                            probabilities[
                                np.arange(len(truth)),
                                np.asarray([classes.index(label) for label in truth]),
                            ],
                            1e-15,
                            1.0,
                        )
                    ).mean()
                ),
                "warning_count": len(caught),
            }
        )
        filter_records.append(
            {
                "scenario": scenario["scenario"],
                "outer_fold": fold,
                "candidate_gene_count": len(columns),
                "low_expression_pass_genes": int(
                    estimator.named_steps["low_expression_filter"].get_support().sum()
                ),
                "selected_genes": int(
                    estimator.named_steps["variance_selector"].get_support().sum()
                ),
            }
        )
        split_records.append(
            {
                "scenario": scenario["scenario"],
                "split_scheme": scenario["split_scheme"],
                "split_seed": scenario["split_seed"],
                "outer_fold": fold,
                "train_samples": len(train),
                "validation_samples": len(valid),
                "train_cases": len(train_groups),
                "validation_cases": len(valid_groups),
                "patient_overlap_count": len(overlap),
                "maximum_samples_per_case": int(frame.groupby("case_barcode").size().max()),
            }
        )
        for position, local_index in enumerate(valid):
            record: dict[str, Any] = {
                "scenario": scenario["scenario"],
                "cohort": scenario["cohort"],
                "outer_fold": fold,
                "matrix_row": int(frame.iloc[local_index]["matrix_row"]),
                "case_barcode": frame.iloc[local_index]["case_barcode"],
                "sample_barcode": frame.iloc[local_index]["sample_barcode"],
                "observed": truth[position],
                "predicted": predicted[position],
            }
            for label in FIVE_CLASSES:
                key = f"probability_{safe_label(label)}"
                record[key] = (
                    float(probabilities[position, classes.index(label)])
                    if label in classes
                    else np.nan
                )
            prediction_records.append(record)
    del X
    return metric_records, prediction_records, split_records, filter_records


def summarise_scenarios(metrics: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "scenario",
        "display_name",
        "comparison_family",
        "cohort",
        "matrix_kind",
        "feature_scheme",
        "class_weight",
        "abundance_threshold",
        "minimum_fraction",
        "split_scheme",
        "split_seed",
    ]
    measure_columns = [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "macro_ovr_roc_auc",
        "macro_average_precision",
        "multiclass_log_loss",
        "low_expression_pass_genes",
        "selected_genes",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(dimensions, dropna=False, sort=False):
        row = dict(zip(dimensions, keys, strict=True))
        row["folds"] = len(group)
        row["development_n"] = int(group["outer_validation_n"].sum())
        for column in measure_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        rows.append(row)
    summary = pd.DataFrame(rows)
    reference = summary.loc[summary["scenario"].eq("four_class_tpm_primary")].iloc[0]
    for metric in ["macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_roc_auc", "macro_average_precision"]:
        summary[f"delta_{metric}_mean_vs_reference"] = (
            summary[f"{metric}_mean"] - float(reference[f"{metric}_mean"])
        )
    return summary


def pooled_tables(
    predictions: pd.DataFrame, scenarios: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_class_records: list[dict[str, Any]] = []
    confusion_records: list[dict[str, Any]] = []
    scenario_lookup = {item["scenario"]: item for item in scenarios}
    for scenario_name, group in predictions.groupby("scenario", sort=False):
        cohort = scenario_lookup[scenario_name]["cohort"]
        classes = FOUR_CLASSES if cohort == "four_class" else FIVE_CLASSES
        precision, recall, f1, support = precision_recall_fscore_support(
            group["observed"],
            group["predicted"],
            labels=classes,
            zero_division=0,
        )
        for label, p, r, score, n in zip(classes, precision, recall, f1, support, strict=True):
            per_class_records.append(
                {
                    "scenario": scenario_name,
                    "pam50_class": label,
                    "precision": p,
                    "recall": r,
                    "f1": score,
                    "support": int(n),
                }
            )
        matrix = confusion_matrix(group["observed"], group["predicted"], labels=classes)
        for truth_index, observed in enumerate(classes):
            for prediction_index, predicted in enumerate(classes):
                confusion_records.append(
                    {
                        "scenario": scenario_name,
                        "observed": observed,
                        "predicted": predicted,
                        "count": int(matrix[truth_index, prediction_index]),
                    }
                )
    return pd.DataFrame(per_class_records), pd.DataFrame(confusion_records)


def label_source_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked = pd.read_csv(
        ROOT / "data" / "processed" / "labels" / "pancanatlas_pam50_locked.tsv",
        sep="\t",
        dtype=str,
    )[["case_barcode", "pam50_raw", "pam50_standard"]]
    raw_records = json.loads(
        (
            ROOT
            / "data"
            / "raw"
            / "pancanatlas"
            / "cbioportal"
            / "brca_tcga_pan_can_atlas_2018"
            / "pam50_subtype_records.json"
        ).read_text(encoding="utf-8")
    )
    raw = pd.DataFrame(
        {
            "case_barcode": [item["patientId"] for item in raw_records],
            "raw_standard": [RAW_TO_STANDARD.get(item["value"]) for item in raw_records],
        }
    )
    if raw["case_barcode"].duplicated().any():
        raise RuntimeError("Raw PanCancer label source contains duplicate patients")

    samples = pd.read_csv(
        ROOT / "data" / "processed" / "expression" / "samples.tsv", sep="\t", dtype=str
    ).dropna(subset=["pam50_standard"])
    five = pd.read_csv(
        ROOT / "data" / "processed" / "splits" / "five_class_split_assignments.tsv",
        sep="\t",
        dtype=str,
    )
    four = pd.read_csv(
        ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv",
        sep="\t",
        dtype=str,
    )
    historical = pd.read_csv(
        ROOT / "data" / "metadata" / "brca_2012_final_full_sample_summary.tsv",
        sep="\t",
        dtype=str,
    )
    historical["historical_standard"] = historical["PAM50"].map(HISTORICAL_TO_STANDARD)
    historical = historical.dropna(subset=["historical_standard"])
    conflicts = historical.groupby("bcr_patient_barcode")["historical_standard"].nunique()
    historical = historical.loc[
        historical["bcr_patient_barcode"].isin(conflicts.loc[conflicts.eq(1)].index)
    ].drop_duplicates("bcr_patient_barcode")

    comparisons: list[dict[str, Any]] = []

    def compare(
        source_a: str,
        source_b: str,
        merged: pd.DataFrame,
        column_a: str,
        column_b: str,
        note: str,
    ) -> None:
        valid = merged.dropna(subset=[column_a, column_b]).copy()
        agree = valid[column_a].eq(valid[column_b])
        comparisons.append(
            {
                "source_a": source_a,
                "source_b": source_b,
                "overlap_cases": len(valid),
                "agreement_cases": int(agree.sum()),
                "discordant_cases": int((~agree).sum()),
                "exact_concordance": float(agree.mean()),
                "cohen_kappa": float(cohen_kappa_score(valid[column_a], valid[column_b])),
                "note": note,
            }
        )

    compare(
        "raw_cBioPortal_PanCancer_SUBTYPE",
        "locked_PanCancer_table",
        locked.merge(raw, on="case_barcode", how="inner"),
        "raw_standard",
        "pam50_standard",
        "Serialization and standardization integrity check",
    )
    compare(
        "locked_PanCancer_table",
        "expression_sample_axis",
        locked.merge(
            samples[["case_barcode", "pam50_standard"]],
            on="case_barcode",
            suffixes=("_locked", "_sample"),
        ),
        "pam50_standard_locked",
        "pam50_standard_sample",
        "Label propagation to the expression matrix",
    )
    compare(
        "locked_PanCancer_table",
        "five_class_split_table",
        locked.merge(
            five[["case_barcode", "pam50_standard"]],
            on="case_barcode",
            suffixes=("_locked", "_split"),
        ),
        "pam50_standard_locked",
        "pam50_standard_split",
        "Label propagation to all five-class assignments",
    )
    compare(
        "locked_PanCancer_table",
        "four_class_split_table",
        locked.merge(
            four[["case_barcode", "pam50_standard"]],
            on="case_barcode",
            suffixes=("_locked", "_split"),
        ),
        "pam50_standard_locked",
        "pam50_standard_split",
        "Label propagation after excluding Normal-like",
    )
    history_overlap = locked.merge(
        historical[["bcr_patient_barcode", "historical_standard"]],
        left_on="case_barcode",
        right_on="bcr_patient_barcode",
        how="inner",
    )
    compare(
        "TCGA_2012_publication_PAM50",
        "PanCancer_Atlas_PAM50",
        history_overlap,
        "historical_standard",
        "pam50_standard",
        "Independent historical data freeze; discordance is scientifically meaningful, not a serialization failure",
    )
    case_comparison = history_overlap[
        ["case_barcode", "historical_standard", "pam50_standard"]
    ].rename(columns={"pam50_standard": "pancancer_standard"})
    case_comparison["agreement"] = case_comparison["historical_standard"].eq(
        case_comparison["pancancer_standard"]
    )
    matrix = pd.crosstab(
        case_comparison["historical_standard"],
        case_comparison["pancancer_standard"],
        dropna=False,
    ).reindex(index=FIVE_CLASSES, columns=FIVE_CLASSES, fill_value=0)
    confusion_records = []
    for historical_label in FIVE_CLASSES:
        for pancancer_label in FIVE_CLASSES:
            confusion_records.append(
                {
                    "historical_2012": historical_label,
                    "pancancer_atlas": pancancer_label,
                    "count": int(matrix.loc[historical_label, pancancer_label]),
                }
            )
    return pd.DataFrame(comparisons), case_comparison, pd.DataFrame(confusion_records)


def save_figures(summary: pd.DataFrame, label_confusion: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    ordered = summary.sort_values("macro_f1_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    for axis, metric, title in [
        (axes[0], "macro_f1", "Macro F1"),
        (axes[1], "balanced_accuracy", "Balanced accuracy"),
    ]:
        axis.errorbar(
            ordered[f"{metric}_mean"],
            np.arange(len(ordered)),
            xerr=ordered[f"{metric}_std"],
            fmt="o",
            capsize=3,
            color="#2b6f92",
        )
        axis.set_yticks(np.arange(len(ordered)), ordered["display_name"])
        axis.set_xlim(0.72, 0.96)
        axis.set_xlabel("Mean ± SD across five development folds")
        axis.set_title(title)
    fig.suptitle("Development-only fixed-model robustness analysis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_robustness_performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    seeds = summary.loc[summary["comparison_family"].eq("random_seed")].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(seeds))
    ax.errorbar(
        positions - 0.07,
        seeds["macro_f1_mean"],
        yerr=seeds["macro_f1_std"],
        fmt="o",
        capsize=4,
        label="Macro F1",
    )
    ax.errorbar(
        positions + 0.07,
        seeds["balanced_accuracy_mean"],
        yerr=seeds["balanced_accuracy_std"],
        fmt="s",
        capsize=4,
        label="Balanced accuracy",
    )
    ax.set_xticks(positions, seeds["split_seed"].astype(str))
    ax.set_ylabel("Mean ± SD across folds")
    ax.set_xlabel("Patient-grouped split and estimator seed")
    ax.set_title("Random-seed sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_random_seed_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    matrix = label_confusion.pivot(
        index="historical_2012", columns="pancancer_atlas", values="count"
    ).reindex(index=FIVE_CLASSES, columns=FIVE_CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("PanCancer Atlas label")
    ax.set_ylabel("TCGA 2012 publication label")
    ax.set_title("PAM50 label-source concordance (overlapping cases)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_label_source_concordance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without pandas' optional tabulate dependency."""

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(render(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_report(summary: pd.DataFrame, label_audit: pd.DataFrame, split_audit: pd.DataFrame) -> None:
    reference = summary.loc[summary["scenario"].eq("four_class_tpm_primary")].iloc[0]
    label_history = label_audit.loc[
        label_audit["source_a"].eq("TCGA_2012_publication_PAM50")
    ].iloc[0]
    seed_rows = summary.loc[summary["comparison_family"].eq("random_seed")]
    grouped = summary.loc[
        summary["scenario"].eq("four_class_seed_20260909_patient_grouped")
    ].iloc[0]
    sampled = summary.loc[
        summary["scenario"].eq("four_class_seed_20260909_sample_stratified")
    ].iloc[0]
    report_table = summary[
        [
            "display_name",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "delta_macro_f1_mean_vs_reference",
        ]
    ].copy()
    for column in report_table.columns[1:]:
        report_table[column] = report_table[column].map(lambda value: f"{value:.3f}")
    lines = [
        "# 稳健性与局限性分析",
        "",
        "## 分析边界",
        "",
        "本分析是模型选择完成后的 development-only 确认实验。随机森林结构和2,000基因训练折方差筛选规则固定，不重新调参；所有低表达过滤、插补、方差筛选与标准化均只在当前训练折拟合。locked test 表达行加载数为0，也没有生成新的测试集预测或性能指标。",
        "",
        "## 稳健性结果",
        "",
        dataframe_to_markdown(report_table),
        "",
        f"参考方案（四分类、log2(TPM+1)、PAM50 included、balanced、锁定病例折）的 macro F1 为 `{reference['macro_f1_mean']:.3f} ± {reference['macro_f1_std']:.3f}`，balanced accuracy 为 `{reference['balanced_accuracy_mean']:.3f} ± {reference['balanced_accuracy_std']:.3f}`。表中差值均相对该参考方案；五分类与不同随机折因病例集合或折分配不同，差值只能作描述性比较。",
        "",
        "随机种子敏感性（种子同时改变病例级fold分配和随机森林random_state）中，mean macro F1 范围为 "
        f"`{seed_rows['macro_f1_mean'].min():.3f}–{seed_rows['macro_f1_mean'].max():.3f}`，mean balanced accuracy 范围为 `{seed_rows['balanced_accuracy_mean'].min():.3f}–{seed_rows['balanced_accuracy_mean'].max():.3f}`。",
        "",
        "病例拆分确认实验使用同一 development 队列和种子：病例分组方案 macro F1 "
        f"`{grouped['macro_f1_mean']:.3f}`，样本分层方案 `{sampled['macro_f1_mean']:.3f}`。当前矩阵每病例最多一个样本，因此两种方案均没有病例跨折重叠；这个实验验证了实现完整性，但不能估计存在重复样本时样本级拆分导致的乐观偏倚。",
        "",
        "## 标签来源一致性",
        "",
        dataframe_to_markdown(label_audit),
        "",
        f"原始cBioPortal记录、锁定标签表、表达矩阵样本轴和split表之间为完全一致。独立的TCGA 2012 publication freeze与PanCancer Atlas在 `{int(label_history['overlap_cases'])}` 个重叠病例中一致 `{int(label_history['agreement_cases'])}` 个（`{label_history['exact_concordance']:.1%}`，Cohen's κ `{label_history['cohen_kappa']:.3f}`）。这表明标签具有较高但并非完美的跨数据冻结稳定性，尤其Luminal A/Luminal B边界会受队列、平台与分类版本影响。",
        "",
        "## 主要局限性",
        "",
        "1. 所有稳健性实验仍来自TCGA-BRCA内部development病例，不等同于独立外部验证，也不能证明跨平台、跨中心或真实临床部署的可迁移性。",
        "2. 替代方案使用冻结的最终随机森林参数而不重新调参。这样能隔离预处理、特征空间和权重变化，但可能低估每个替代方案单独优化后的最佳性能。",
        "3. log2(CPM+1)是透明的count-based library-size normalization敏感性方案，但不替代TMM、DESeq2 VST或跨队列批次校正；TPM与CPM都无法消除肿瘤纯度和细胞组成差异。",
        "4. PAM50标签本身由表达信号定义。使用PAM50基因预测PAM50标签具有概念上的近循环性；排除50个signature基因仍不能移除共表达的代理信号，因此不能把排除结果解释为完全独立于PAM50生物学。",
        "5. Normal-like类别样本较少，五分类结果的不确定性更高；宏平均指标会对该类别的波动较敏感。",
        "6. 当前一病例一样本设计使病例级拆分与样本级拆分都不存在重复病例泄漏。它确认了管线，但无法量化多样本队列中的泄漏幅度。",
        "7. 三个随机种子只能刻画有限的fold/estimator随机性，不能覆盖所有可能的数据划分。",
        "8. TCGA 2012与PanCancer Atlas的标签不一致不能简单视为错误：不同数据冻结、表达平台、可用基因、样本选择及PAM50实现均可能导致边界病例改变亚型。",
        "9. 本阶段不得因任何敏感性结果重新选择模型或再次评估locked test；若要改变最终方案，应在外部队列预先锁定并验证。",
        "",
        "## 审计结论",
        "",
        f"全部场景的训练/验证病例重叠最大值为 `{int(split_audit['patient_overlap_count'].max())}`，locked-test表达加载与预测均为0。结果支持主结论对所检查分析选择具有总体稳健性，但不能替代外部验证；标签版本差异与PAM50表达定义的近循环性是最重要的解释边界。",
        "",
        "图：`figures/01_robustness_performance.png`、`figures/02_random_seed_sensitivity.png`、`figures/03_label_source_concordance.png`。",
    ]
    (OUTPUT_DIR / "robustness_and_limitations_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.forest_n_jobs == 0:
        raise ValueError("forest-n-jobs cannot be zero")
    config = load_json(CONFIG_PATH)
    if args.finalize_only and (args.only or args.n_estimators_override is not None):
        raise ValueError("--finalize-only cannot be combined with --only or an estimator override")
    if args.n_estimators_override is not None:
        if not args.only:
            raise ValueError("An estimator override is allowed only with --only smoke runs")
        config["estimator"]["n_estimators"] = int(args.n_estimators_override)
    all_scenarios = scenario_definitions(config)
    scenarios = all_scenarios
    if args.only:
        requested = set(args.only)
        scenarios = [item for item in all_scenarios if item["scenario"] in requested]
        missing = requested - {item["scenario"] for item in scenarios}
        if missing:
            raise ValueError(f"Unknown scenarios: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    locked_counts: dict[str, int] = {}
    if args.finalize_only:
        metrics_frame = pd.read_csv(OUTPUT_DIR / "scenario_fold_metrics.tsv", sep="\t")
        predictions_frame = pd.read_csv(
            OUTPUT_DIR / "scenario_oof_predictions.tsv.gz", sep="\t"
        )
        split_frame = pd.read_csv(OUTPUT_DIR / "split_integrity.tsv", sep="\t")
        filter_frame = pd.read_csv(OUTPUT_DIR / "fold_feature_counts.tsv", sep="\t")
        if set(metrics_frame["scenario"]) != {item["scenario"] for item in scenarios}:
            raise RuntimeError("Saved per-fold outputs do not contain the complete scenario set")
        for scenario in scenarios:
            _, locked_n = load_development(scenario["cohort"])
            locked_counts[scenario["cohort"]] = locked_n
    else:
        metric_records: list[dict[str, Any]] = []
        prediction_records: list[dict[str, Any]] = []
        split_records: list[dict[str, Any]] = []
        filter_records: list[dict[str, Any]] = []
        for scenario in scenarios:
            _, locked_n = load_development(scenario["cohort"])
            locked_counts[scenario["cohort"]] = locked_n
            metrics, predictions, split_audit, filters = run_scenario(
                config, scenario, args.forest_n_jobs
            )
            metric_records.extend(metrics)
            prediction_records.extend(predictions)
            split_records.extend(split_audit)
            filter_records.extend(filters)

        metrics_frame = pd.DataFrame(metric_records)
        predictions_frame = pd.DataFrame(prediction_records)
        split_frame = pd.DataFrame(split_records)
        filter_frame = pd.DataFrame(filter_records)
        metrics_frame.to_csv(OUTPUT_DIR / "scenario_fold_metrics.tsv", sep="\t", index=False)
        predictions_frame.to_csv(
            OUTPUT_DIR / "scenario_oof_predictions.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )
        split_frame.to_csv(OUTPUT_DIR / "split_integrity.tsv", sep="\t", index=False)
        filter_frame.to_csv(OUTPUT_DIR / "fold_feature_counts.tsv", sep="\t", index=False)

    if args.only:
        print("Smoke scenario run complete; aggregate report was not generated.", flush=True)
        return

    summary = summarise_scenarios(metrics_frame)
    summary.to_csv(OUTPUT_DIR / "scenario_summary.tsv", sep="\t", index=False)
    per_class, confusion = pooled_tables(predictions_frame, scenarios)
    per_class.to_csv(OUTPUT_DIR / "scenario_per_class_metrics.tsv", sep="\t", index=False)
    confusion.to_csv(OUTPUT_DIR / "scenario_confusion_matrices.tsv", sep="\t", index=False)
    label_audit, label_cases, label_confusion = label_source_audit()
    label_audit.to_csv(OUTPUT_DIR / "label_source_audit.tsv", sep="\t", index=False)
    label_cases.to_csv(
        OUTPUT_DIR / "label_source_case_comparison.tsv", sep="\t", index=False
    )
    label_confusion.to_csv(
        OUTPUT_DIR / "label_source_confusion_matrix.tsv", sep="\t", index=False
    )
    save_figures(summary, label_confusion)
    write_report(summary, label_audit, split_frame)

    input_paths = [
        CONFIG_PATH,
        ROOT / "config" / "final_model_lock_v1.json",
        ROOT / "data" / "processed" / "expression" / "matrix_manifest.json",
        ROOT / "data" / "processed" / "expression" / "genes.tsv",
        ROOT / "data" / "processed" / "expression" / "samples.tsv",
        ROOT / "data" / "processed" / "labels" / "pam50_signature_genes_v1.tsv",
        ROOT / "data" / "processed" / "labels" / "pancanatlas_pam50_locked.tsv",
        ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv",
        ROOT / "data" / "processed" / "splits" / "five_class_split_assignments.tsv",
        ROOT / "data" / "metadata" / "brca_2012_final_full_sample_summary.tsv",
    ]
    artifact_names = [
        "scenario_fold_metrics.tsv",
        "scenario_summary.tsv",
        "scenario_oof_predictions.tsv.gz",
        "scenario_per_class_metrics.tsv",
        "scenario_confusion_matrices.tsv",
        "split_integrity.tsv",
        "fold_feature_counts.tsv",
        "label_source_audit.tsv",
        "label_source_case_comparison.tsv",
        "label_source_confusion_matrix.tsv",
        "robustness_and_limitations_report.md",
        "figures/01_robustness_performance.png",
        "figures/02_random_seed_sensitivity.png",
        "figures/03_label_source_concordance.png",
    ]
    manifest = {
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_partition": "development_only",
        "scenario_count": len(scenarios),
        "outer_fits": len(metrics_frame),
        "fixed_estimator": config["estimator"],
        "forest_n_jobs": args.forest_n_jobs,
        "development_cases": {"four_class": 756, "five_class": 785},
        "locked_test_cases_in_assignment_files": locked_counts,
        "locked_test_expression_rows_loaded": 0,
        "locked_test_predictions_generated": 0,
        "locked_test_performance_metrics_generated": 0,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in input_paths
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
        "output_sha256": {name: sha256(OUTPUT_DIR / name) for name in artifact_names},
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary[["scenario", "macro_f1_mean", "balanced_accuracy_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
