#!/usr/bin/env python3
"""Run leakage-audited global, class-level, and individual interpretation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from joblib import parallel_backend
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score

try:
    from evaluation_framework import build_pipeline, load_config
except ModuleNotFoundError:  # pragma: no cover
    from scripts.evaluation_framework import build_pipeline, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "interpretability_v1.json"
EVALUATION_PATH = ROOT / "config" / "evaluation.json"
FOREST_PATH = ROOT / "config" / "random_forest_v1.json"
ASSIGNMENT_PATH = (
    ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv"
)
CASE_SELECTION_PATH = (
    ROOT / "data" / "processed" / "interpretability" / "shap_case_selection.tsv"
)
BACKGROUND_SELECTION_PATH = (
    ROOT / "data" / "processed" / "interpretability" / "shap_background_selection.tsv"
)
INTERPRETATION_ACCESS_PATH = (
    ROOT / "data" / "processed" / "splits" / "interpretation_test_access_v1.json"
)
GENE_PATH = ROOT / "data" / "processed" / "expression" / "genes.tsv"
COEFFICIENT_PATH = (
    ROOT / "outputs" / "modeling" / "logistic_comparison" / "outer_fold_coefficients.tsv.gz"
)
FOREST_PARAMETER_PATH = (
    ROOT / "outputs" / "modeling" / "random_forest" / "best_hyperparameters.tsv"
)
FINAL_PARAMETER_PATH = ROOT / "outputs" / "final_evaluation" / "selected_hyperparameters.json"
FINAL_PREDICTION_PATH = ROOT / "outputs" / "final_evaluation" / "locked_test_predictions.tsv"
OUTPUT_DIR = ROOT / "outputs" / "interpretability"


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


def verify_lock(config: dict[str, Any]) -> None:
    if shap.__version__ != config["shap"]["version"]:
        raise RuntimeError(
            f"SHAP version mismatch: {shap.__version__} != {config['shap']['version']}"
        )
    for relative, expected in config["inputs_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Interpretability input hash mismatch: {relative}")
    for relative, expected in config["locked_artifacts"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Interpretability lock artifact mismatch: {relative}")
    access = json.loads(
        (ROOT / "data/processed/splits/final_test_access_v1.json").read_text(
            encoding="utf-8"
        )
    )
    if access.get("status") != "COMPLETE" or access.get("evaluation_attempt") != 1:
        raise RuntimeError("Final performance evaluation audit is not complete")


def selected_gene_frame(
    pipeline: Any, candidate_indices: np.ndarray, genes: pd.DataFrame
) -> pd.DataFrame:
    low = pipeline.named_steps["low_expression_filter"].get_support(indices=True)
    variance = pipeline.named_steps["variance_selector"].get_support(indices=True)
    matrix_columns = candidate_indices[low[variance]]
    frame = genes.set_index("matrix_column").loc[matrix_columns].reset_index()
    frame.insert(0, "model_feature_index", np.arange(len(frame)))
    return frame


def summarize_elastic_net(
    genes: pd.DataFrame, config: dict[str, Any]
) -> list[str]:
    print(f"[{now_utc()}] Aggregating saved Elastic-net coefficients", flush=True)
    model = config["elastic_net"]["model"]
    folds = int(config["elastic_net"]["outer_folds"])
    coefficient = pd.read_csv(COEFFICIENT_PATH, sep="\t")
    coefficient = coefficient.loc[coefficient["model"].eq(model)].copy()
    if coefficient["outer_fold"].nunique() != folds:
        raise RuntimeError("Elastic-net coefficient file does not contain five folds")
    coding = genes.loc[genes["is_protein_coding"].astype(bool)].copy()
    classes = list(config["class_order"])
    threshold = 1e-12

    grouped = coefficient.groupby(["pam50_class", "matrix_column"], as_index=False).agg(
        coefficient_sum=("coefficient", "sum"),
        coefficient_sum_squares=("coefficient", lambda values: float(np.square(values).sum())),
        absolute_coefficient_sum=("absolute_coefficient", "sum"),
        selected_fold_count=("outer_fold", "nunique"),
        nonzero_fold_count=("is_nonzero", "sum"),
        positive_fold_count=("coefficient", lambda values: int((values > threshold).sum())),
        negative_fold_count=("coefficient", lambda values: int((values < -threshold).sum())),
    )
    index = pd.MultiIndex.from_product(
        [classes, coding["matrix_column"].astype(int)],
        names=["pam50_class", "matrix_column"],
    ).to_frame(index=False)
    summary = index.merge(grouped, on=["pam50_class", "matrix_column"], how="left")
    numeric = [
        "coefficient_sum",
        "coefficient_sum_squares",
        "absolute_coefficient_sum",
        "selected_fold_count",
        "nonzero_fold_count",
        "positive_fold_count",
        "negative_fold_count",
    ]
    summary[numeric] = summary[numeric].fillna(0)
    summary["mean_coefficient_including_zeros"] = summary["coefficient_sum"] / folds
    summary["mean_absolute_coefficient_including_zeros"] = (
        summary["absolute_coefficient_sum"] / folds
    )
    mean = summary["mean_coefficient_including_zeros"]
    variance = (summary["coefficient_sum_squares"] - folds * mean.pow(2)) / (folds - 1)
    summary["coefficient_sd_including_zeros"] = np.sqrt(variance.clip(lower=0))
    nonzero = summary["nonzero_fold_count"].replace(0, np.nan)
    summary["sign_consistency"] = (
        summary[["positive_fold_count", "negative_fold_count"]].max(axis=1) / nonzero
    ).fillna(0.0)
    summary = summary.merge(
        coding[
            [
                "matrix_column",
                "gene_id",
                "gene_id_without_version",
                "gene_name",
                "gene_type",
            ]
        ],
        on="matrix_column",
        how="left",
        validate="many_to_one",
    )

    rank_records: list[dict[str, Any]] = []
    for (fold, subtype), group in coefficient.groupby(["outer_fold", "pam50_class"]):
        nonzero_group = group.loc[group["coefficient"].abs().gt(threshold)].copy()
        nonzero_group["absolute_rank"] = nonzero_group["absolute_coefficient"].rank(
            method="first", ascending=False
        )
        positives = nonzero_group.loc[nonzero_group["coefficient"].gt(0)].copy()
        positives["direction_rank"] = positives["coefficient"].rank(
            method="first", ascending=False
        )
        negatives = nonzero_group.loc[nonzero_group["coefficient"].lt(0)].copy()
        negatives["direction_rank"] = negatives["coefficient"].rank(
            method="first", ascending=True
        )
        direction_lookup = {
            int(row.matrix_column): ("positive", int(row.direction_rank))
            for row in positives.itertuples()
        }
        direction_lookup.update(
            {
                int(row.matrix_column): ("negative", int(row.direction_rank))
                for row in negatives.itertuples()
            }
        )
        for row in nonzero_group.itertuples():
            direction, direction_rank = direction_lookup[int(row.matrix_column)]
            rank_records.append(
                {
                    "outer_fold": int(fold),
                    "pam50_class": subtype,
                    "matrix_column": int(row.matrix_column),
                    "gene_name": row.gene_name,
                    "coefficient": float(row.coefficient),
                    "direction": direction,
                    "absolute_rank": int(row.absolute_rank),
                    "direction_rank": direction_rank,
                }
            )
    ranks = pd.DataFrame(rank_records)
    for cutoff in config["elastic_net"]["stability_cutoffs"]:
        ranks[f"in_absolute_top_{cutoff}"] = ranks["absolute_rank"].le(int(cutoff))
        ranks[f"in_direction_top_{cutoff}"] = ranks["direction_rank"].le(int(cutoff))
    ranks.to_csv(
        OUTPUT_DIR / "elastic_net_outer_fold_feature_ranks.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    primary = int(config["elastic_net"]["primary_stability_cutoff"])
    top = ranks.loc[ranks[f"in_absolute_top_{primary}"]].copy()
    stability = top.groupby(
        ["pam50_class", "matrix_column", "gene_name"], as_index=False
    ).agg(
        top20_outer_fold_count=("outer_fold", "nunique"),
        mean_coefficient_when_top20=("coefficient", "mean"),
        positive_top20_fold_count=("coefficient", lambda values: int((values > 0).sum())),
        negative_top20_fold_count=("coefficient", lambda values: int((values < 0).sum())),
    )
    stability["top20_outer_fold_frequency"] = stability["top20_outer_fold_count"] / folds
    stability["dominant_direction"] = np.where(
        stability["positive_top20_fold_count"] >= stability["negative_top20_fold_count"],
        "positive",
        "negative",
    )
    stability = stability.sort_values(
        ["top20_outer_fold_count", "pam50_class", "gene_name"],
        ascending=[False, True, True],
    )
    stability.to_csv(
        OUTPUT_DIR / "elastic_net_top20_stability.tsv", sep="\t", index=False
    )
    summary = summary.merge(
        stability[
            [
                "pam50_class",
                "matrix_column",
                "top20_outer_fold_count",
                "top20_outer_fold_frequency",
            ]
        ],
        on=["pam50_class", "matrix_column"],
        how="left",
    )
    summary[["top20_outer_fold_count", "top20_outer_fold_frequency"]] = summary[
        ["top20_outer_fold_count", "top20_outer_fold_frequency"]
    ].fillna(0)
    summary.to_csv(
        OUTPUT_DIR / "elastic_net_coefficient_summary.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    class_top: list[pd.DataFrame] = []
    n = int(config["elastic_net"]["class_top_positive"])
    for subtype in classes:
        group = summary.loc[summary["pam50_class"].eq(subtype)]
        positive = group.loc[group["mean_coefficient_including_zeros"].gt(0)].nlargest(
            n, "mean_coefficient_including_zeros"
        )
        negative = group.loc[group["mean_coefficient_including_zeros"].lt(0)].nsmallest(
            n, "mean_coefficient_including_zeros"
        )
        for direction, selected in (("positive", positive), ("negative", negative)):
            selected = selected.copy()
            selected["direction"] = direction
            selected["direction_rank"] = np.arange(1, len(selected) + 1)
            class_top.append(selected)
    class_top_frame = pd.concat(class_top, ignore_index=True)
    class_top_frame.to_csv(
        OUTPUT_DIR / "elastic_net_class_top_genes.tsv", sep="\t", index=False
    )

    global_summary = summary.groupby(
        ["matrix_column", "gene_id", "gene_id_without_version", "gene_name"],
        as_index=False,
    ).agg(
        mean_absolute_coefficient_across_classes=(
            "mean_absolute_coefficient_including_zeros",
            "mean",
        ),
        max_absolute_mean_coefficient=(
            "mean_coefficient_including_zeros",
            lambda values: float(np.abs(values).max()),
        ),
        maximum_class_top20_frequency=("top20_outer_fold_frequency", "max"),
    )
    dominant = summary.loc[
        summary.groupby("matrix_column")["mean_coefficient_including_zeros"]
        .apply(lambda values: values.abs().idxmax())
        .to_numpy(),
        ["matrix_column", "pam50_class", "mean_coefficient_including_zeros"],
    ].rename(
        columns={
            "pam50_class": "dominant_class",
            "mean_coefficient_including_zeros": "dominant_class_mean_coefficient",
        }
    )
    global_summary = global_summary.merge(dominant, on="matrix_column", how="left")
    global_summary = global_summary.sort_values(
        "mean_absolute_coefficient_across_classes", ascending=False
    )
    global_summary.to_csv(
        OUTPUT_DIR / "elastic_net_global_coefficients.tsv", sep="\t", index=False
    )
    return [
        "elastic_net_outer_fold_feature_ranks.tsv.gz",
        "elastic_net_top20_stability.tsv",
        "elastic_net_coefficient_summary.tsv.gz",
        "elastic_net_class_top_genes.tsv",
        "elastic_net_global_coefficients.tsv",
    ]


def run_permutation_importance(
    matrix: np.ndarray,
    candidate_indices: np.ndarray,
    genes: pd.DataFrame,
    assignments: pd.DataFrame,
    evaluation: dict[str, Any],
    forest: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    if len(development) != 756:
        raise RuntimeError("Expected 756 development cases")
    if assignments.loc[assignments["holdout_split"].eq("locked_test"), "outer_fold"].notna().any():
        raise RuntimeError("Locked-test cases unexpectedly have outer-fold assignments")
    rows = development["matrix_row"].astype(int).to_numpy()
    X = np.asarray(matrix[rows][:, candidate_indices], dtype=np.float32)
    y = development[evaluation["label_column"]].astype(str).to_numpy()
    fold_values = development["outer_fold"].astype(int).to_numpy()
    parameters = pd.read_csv(FOREST_PARAMETER_PATH, sep="\t").set_index("outer_fold")
    records: list[dict[str, Any]] = []
    repeat_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    background_records: list[dict[str, Any]] = []

    for fold in range(1, int(config["permutation_importance"]["outer_folds"]) + 1):
        print(f"[{now_utc()}] Held-out permutation importance fold {fold}/5", flush=True)
        train_index = np.flatnonzero(fold_values != fold)
        valid_index = np.flatnonzero(fold_values == fold)
        if np.intersect1d(train_index, valid_index).size:
            raise RuntimeError("Outer train/validation indices overlap")
        best_params = json.loads(parameters.loc[fold, "best_parameters_json"])
        pipeline = build_pipeline(
            "random_forest", evaluation, random_seed=int(forest["random_seed"]) + fold
        )
        pipeline.set_params(
            classifier__class_weight="balanced",
            classifier__criterion="gini",
            classifier__bootstrap=True,
            classifier__n_jobs=int(config["permutation_importance"]["forest_n_jobs"]),
            **best_params,
        )
        pipeline.fit(X[train_index], y[train_index])
        selected = selected_gene_frame(pipeline, candidate_indices, genes)
        low_local = pipeline.named_steps["low_expression_filter"].get_support(
            indices=True
        )
        low_columns = candidate_indices[low_local]
        low_genes = genes.set_index("matrix_column").loc[low_columns].reset_index()
        for gene in low_genes.itertuples():
            background_records.append(
                {
                    "outer_fold": fold,
                    "matrix_column": int(gene.matrix_column),
                    "gene_id": gene.gene_id,
                    "gene_id_without_version": gene.gene_id_without_version,
                    "gene_name": gene.gene_name,
                    "passed_low_expression_filter": True,
                }
            )
        transformed_valid = pipeline[:-1].transform(X[valid_index])
        classifier = pipeline.named_steps["classifier"]
        predicted = classifier.predict(transformed_valid)
        baseline_macro_f1 = float(f1_score(y[valid_index], predicted, average="macro"))
        baseline_balanced = float(balanced_accuracy_score(y[valid_index], predicted))
        with parallel_backend(
            "threading",
            n_jobs=int(config["permutation_importance"]["permutation_n_jobs"]),
        ):
            result = permutation_importance(
                classifier,
                transformed_valid,
                y[valid_index],
                scoring={
                    "macro_f1": "f1_macro",
                    "balanced_accuracy": "balanced_accuracy",
                },
                n_repeats=int(config["permutation_importance"]["repeats"]),
                random_state=int(config["random_seed"]) + 1000 + fold,
                n_jobs=int(config["permutation_importance"]["permutation_n_jobs"]),
            )
        if set(result) != {"macro_f1", "balanced_accuracy"}:
            raise RuntimeError("Unexpected permutation scorer result")
        for feature_index, gene in selected.iterrows():
            record = {
                "outer_fold": fold,
                "model_feature_index": int(feature_index),
                "matrix_column": int(gene["matrix_column"]),
                "gene_id": gene["gene_id"],
                "gene_id_without_version": gene["gene_id_without_version"],
                "gene_name": gene["gene_name"],
                "macro_f1_importance_mean": float(result["macro_f1"].importances_mean[feature_index]),
                "macro_f1_importance_sd": float(result["macro_f1"].importances_std[feature_index]),
                "balanced_accuracy_importance_mean": float(
                    result["balanced_accuracy"].importances_mean[feature_index]
                ),
                "balanced_accuracy_importance_sd": float(
                    result["balanced_accuracy"].importances_std[feature_index]
                ),
            }
            records.append(record)
            for repeat in range(int(config["permutation_importance"]["repeats"])):
                repeat_records.append(
                    {
                        "outer_fold": fold,
                        "permutation_repeat": repeat + 1,
                        "matrix_column": int(gene["matrix_column"]),
                        "gene_name": gene["gene_name"],
                        "macro_f1_importance": float(
                            result["macro_f1"].importances[feature_index, repeat]
                        ),
                        "balanced_accuracy_importance": float(
                            result["balanced_accuracy"].importances[feature_index, repeat]
                        ),
                    }
                )
        audit_records.append(
            {
                "outer_fold": fold,
                "fit_partition": "outer_train",
                "evaluation_partition": "outer_validation",
                "outer_train_n": len(train_index),
                "outer_validation_n": len(valid_index),
                "outer_train_case_sha256": hashlib.sha256(
                    "\n".join(sorted(development.iloc[train_index]["case_barcode"])).encode()
                ).hexdigest(),
                "outer_validation_case_sha256": hashlib.sha256(
                    "\n".join(sorted(development.iloc[valid_index]["case_barcode"])).encode()
                ).hexdigest(),
                "case_overlap_n": 0,
                "selected_gene_n": len(selected),
                "repeats": int(config["permutation_importance"]["repeats"]),
                "baseline_macro_f1": baseline_macro_f1,
                "baseline_balanced_accuracy": baseline_balanced,
                "locked_test_expression_rows_loaded": 0,
            }
        )

    per_fold = pd.DataFrame(records)
    for cutoff in config["permutation_importance"]["stability_cutoffs"]:
        per_fold[f"macro_f1_rank"] = per_fold.groupby("outer_fold")[
            "macro_f1_importance_mean"
        ].rank(method="first", ascending=False)
        per_fold[f"in_macro_f1_top_{cutoff}"] = per_fold["macro_f1_rank"].le(
            int(cutoff)
        )
    per_fold.to_csv(
        OUTPUT_DIR / "rf_outer_validation_permutation_importance.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(repeat_records).to_csv(
        OUTPUT_DIR / "rf_outer_validation_permutation_repeats.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(audit_records).to_csv(
        OUTPUT_DIR / "rf_permutation_partition_audit.tsv", sep="\t", index=False
    )
    background_by_fold = pd.DataFrame(background_records)
    background_by_fold.to_csv(
        OUTPUT_DIR / "expression_filter_background_by_fold.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    enrichment_background = background_by_fold.groupby(
        ["matrix_column", "gene_id", "gene_id_without_version", "gene_name"],
        as_index=False,
    ).agg(pass_outer_fold_count=("outer_fold", "nunique"))
    enrichment_background["in_union_background"] = enrichment_background[
        "pass_outer_fold_count"
    ].ge(1)
    enrichment_background["in_all_five_folds_background"] = enrichment_background[
        "pass_outer_fold_count"
    ].eq(int(config["permutation_importance"]["outer_folds"]))
    enrichment_background.to_csv(
        OUTPUT_DIR / "enrichment_background.tsv", sep="\t", index=False
    )

    folds = int(config["permutation_importance"]["outer_folds"])
    grouped = per_fold.groupby(
        ["matrix_column", "gene_id", "gene_id_without_version", "gene_name"],
        as_index=False,
    ).agg(
        selected_fold_count=("outer_fold", "nunique"),
        macro_f1_importance_sum=("macro_f1_importance_mean", "sum"),
        macro_f1_importance_mean_when_selected=("macro_f1_importance_mean", "mean"),
        macro_f1_importance_sd_across_selected_folds=("macro_f1_importance_mean", "std"),
        balanced_accuracy_importance_sum=("balanced_accuracy_importance_mean", "sum"),
        balanced_accuracy_importance_mean_when_selected=(
            "balanced_accuracy_importance_mean",
            "mean",
        ),
    )
    grouped["macro_f1_importance_mean_including_unselected_zeros"] = (
        grouped["macro_f1_importance_sum"] / folds
    )
    grouped["balanced_accuracy_importance_mean_including_unselected_zeros"] = (
        grouped["balanced_accuracy_importance_sum"] / folds
    )
    primary = int(config["permutation_importance"]["primary_stability_cutoff"])
    stable = per_fold.loc[per_fold[f"in_macro_f1_top_{primary}"]].groupby(
        ["matrix_column", "gene_name"], as_index=False
    ).agg(top20_outer_fold_count=("outer_fold", "nunique"))
    stable["top20_outer_fold_frequency"] = stable["top20_outer_fold_count"] / folds
    stable = stable.sort_values(
        ["top20_outer_fold_count", "gene_name"], ascending=[False, True]
    )
    stable.to_csv(OUTPUT_DIR / "rf_top20_stability.tsv", sep="\t", index=False)
    grouped = grouped.merge(stable, on=["matrix_column", "gene_name"], how="left")
    grouped[["top20_outer_fold_count", "top20_outer_fold_frequency"]] = grouped[
        ["top20_outer_fold_count", "top20_outer_fold_frequency"]
    ].fillna(0)
    grouped = grouped.sort_values(
        "macro_f1_importance_mean_including_unselected_zeros", ascending=False
    )
    grouped.to_csv(
        OUTPUT_DIR / "rf_permutation_importance_summary.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    return [
        "rf_outer_validation_permutation_importance.tsv.gz",
        "rf_outer_validation_permutation_repeats.tsv.gz",
        "rf_permutation_partition_audit.tsv",
        "expression_filter_background_by_fold.tsv.gz",
        "enrichment_background.tsv",
        "rf_top20_stability.tsv",
        "rf_permutation_importance_summary.tsv.gz",
    ]


def normalize_shap_output(
    explanation: Any, n_samples: int, n_features: int, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(explanation.values, dtype=float)
    if values.shape == (n_samples, n_features, n_classes):
        pass
    elif values.shape == (n_samples, n_classes, n_features):
        values = np.moveaxis(values, 1, 2)
    else:
        raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")
    base = np.asarray(explanation.base_values, dtype=float)
    if base.shape == (n_classes,):
        base = np.tile(base, (n_samples, 1))
    if base.shape != (n_samples, n_classes):
        raise RuntimeError(f"Unexpected SHAP base-value shape: {base.shape}")
    return values, base


def run_shap(
    matrix: np.ndarray,
    candidate_indices: np.ndarray,
    genes: pd.DataFrame,
    assignments: pd.DataFrame,
    evaluation: dict[str, Any],
    forest: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    print(f"[{now_utc()}] Refitting frozen final forest on development only", flush=True)
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    development_rows = development["matrix_row"].astype(int).to_numpy()
    X_development = np.asarray(
        matrix[development_rows][:, candidate_indices], dtype=np.float32
    )
    y_development = development[evaluation["label_column"]].astype(str).to_numpy()
    parameters = json.loads(FINAL_PARAMETER_PATH.read_text(encoding="utf-8"))[
        "best_parameters"
    ]
    pipeline = build_pipeline(
        "random_forest", evaluation, random_seed=int(forest["random_seed"])
    )
    pipeline.set_params(
        classifier__class_weight="balanced",
        classifier__criterion="gini",
        classifier__bootstrap=True,
        classifier__n_jobs=1,
        **parameters,
    )
    pipeline.fit(X_development, y_development)
    selected_genes = selected_gene_frame(pipeline, candidate_indices, genes)
    if len(selected_genes) != 2000:
        raise RuntimeError("Frozen final model did not select 2,000 genes")

    background = pd.read_csv(BACKGROUND_SELECTION_PATH, sep="\t")
    if not background["holdout_split"].eq("development").all():
        raise RuntimeError("SHAP background includes a non-development case")
    if not set(background["matrix_row"]).issubset(set(development_rows)):
        raise RuntimeError("SHAP background rows are not a subset of development")
    background_rows = background["matrix_row"].astype(int).to_numpy()
    row_to_position = {row: position for position, row in enumerate(development_rows)}
    background_positions = np.asarray([row_to_position[row] for row in background_rows])
    X_background = pipeline[:-1].transform(X_development[background_positions])

    cases = pd.read_csv(CASE_SELECTION_PATH, sep="\t")
    if len(cases) != int(config["shap"]["selected_test_case_n"]):
        raise RuntimeError("Unexpected SHAP case count")
    test_rows = set(
        assignments.loc[
            assignments["holdout_split"].eq("locked_test"), "matrix_row"
        ].astype(int)
    )
    if not set(cases["matrix_row"].astype(int)).issubset(test_rows):
        raise RuntimeError("SHAP cases are not locked-test cases")
    if INTERPRETATION_ACCESS_PATH.exists():
        raise RuntimeError("Interpretation test-access record already exists")
    access = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "started_at_utc": now_utc(),
        "purpose": config["shap"]["purpose"],
        "interpretation_config_sha256": sha256(CONFIG_PATH),
        "selected_case_file_sha256": sha256(CASE_SELECTION_PATH),
        "selected_locked_test_case_n": len(cases),
        "performance_metrics_computed": 0,
        "model_selection_performed": False,
    }
    INTERPRETATION_ACCESS_PATH.write_text(
        json.dumps(access, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[{now_utc()}] Loading six preselected test cases for SHAP only", flush=True)
    case_rows = cases["matrix_row"].astype(int).to_numpy()
    X_cases_raw = np.asarray(matrix[case_rows][:, candidate_indices], dtype=np.float32)
    X_cases = pipeline[:-1].transform(X_cases_raw)
    classifier = pipeline.named_steps["classifier"]
    probabilities = classifier.predict_proba(X_cases)
    predicted = classifier.predict(X_cases)
    if not np.array_equal(predicted, cases["predicted"].astype(str).to_numpy()):
        raise RuntimeError("Refitted final model predictions do not match saved predictions")
    saved = pd.read_csv(FINAL_PREDICTION_PATH, sep="\t").set_index("matrix_row")
    for class_index, subtype in enumerate(classifier.classes_):
        saved_values = saved.loc[
            case_rows, f"probability_{safe_label(str(subtype))}"
        ].to_numpy(dtype=float)
        if not np.allclose(probabilities[:, class_index], saved_values, atol=1e-12):
            raise RuntimeError("Refitted final probabilities do not match saved predictions")

    print(f"[{now_utc()}] Computing TreeSHAP probability attributions", flush=True)
    explainer = shap.TreeExplainer(
        classifier,
        data=np.asarray(X_background, dtype=np.float32),
        feature_perturbation=config["shap"]["feature_perturbation"],
        model_output=config["shap"]["model_output"],
    )
    explanation = explainer(np.asarray(X_cases, dtype=np.float32), check_additivity=False)
    values, base_values = normalize_shap_output(
        explanation, len(cases), X_cases.shape[1], len(classifier.classes_)
    )
    raw_selected = pipeline.named_steps["variance_selector"].transform(
        pipeline.named_steps["median_imputer"].transform(
            pipeline.named_steps["low_expression_filter"].transform(X_cases_raw)
        )
    )
    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case_index, case in cases.iterrows():
        for class_index, subtype in enumerate(classifier.classes_):
            reconstructed = float(base_values[case_index, class_index] + values[case_index, :, class_index].sum())
            for feature_index, gene in selected_genes.iterrows():
                rows.append(
                    {
                        "display_order": int(case["display_order"]),
                        "selection_role": case["selection_role"],
                        "matrix_row": int(case["matrix_row"]),
                        "case_barcode": case["case_barcode"],
                        "sample_barcode": case["sample_barcode"],
                        "observed": case["observed"],
                        "predicted": case["predicted"],
                        "explained_class": str(subtype),
                        "model_feature_index": int(feature_index),
                        "matrix_column": int(gene["matrix_column"]),
                        "gene_id": gene["gene_id"],
                        "gene_id_without_version": gene["gene_id_without_version"],
                        "gene_name": gene["gene_name"],
                        "raw_log2_tpm": float(raw_selected[case_index, feature_index]),
                        "scaled_feature_value": float(X_cases[case_index, feature_index]),
                        "shap_value": float(values[case_index, feature_index, class_index]),
                        "base_probability": float(base_values[case_index, class_index]),
                        "model_probability": float(probabilities[case_index, class_index]),
                        "reconstructed_probability": reconstructed,
                    }
                )
        predicted_class_index = int(np.where(classifier.classes_ == case["predicted"])[0][0])
        reconstructed_predicted = float(
            base_values[case_index, predicted_class_index]
            + values[case_index, :, predicted_class_index].sum()
        )
        case_summaries.append(
            {
                "display_order": int(case["display_order"]),
                "selection_role": case["selection_role"],
                "matrix_row": int(case["matrix_row"]),
                "case_barcode": case["case_barcode"],
                "observed": case["observed"],
                "predicted": case["predicted"],
                "correct": bool(case["observed"] == case["predicted"]),
                "confidence": float(case["confidence"]),
                "probability_margin": float(case["probability_margin"]),
                "predicted_probability_refit": float(
                    probabilities[case_index, predicted_class_index]
                ),
                "base_probability": float(base_values[case_index, predicted_class_index]),
                "reconstructed_probability": reconstructed_predicted,
                "absolute_additivity_residual": abs(
                    reconstructed_predicted - probabilities[case_index, predicted_class_index]
                ),
            }
        )
    shap_values = pd.DataFrame(rows)
    shap_values.to_csv(
        OUTPUT_DIR / "shap_values_selected_cases.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    case_summary = pd.DataFrame(case_summaries)
    case_summary.to_csv(OUTPUT_DIR / "shap_case_summary.tsv", sep="\t", index=False)
    predicted_class_values = shap_values.loc[
        shap_values["explained_class"].eq(shap_values["predicted"])
    ].copy()
    predicted_class_values["absolute_shap_value"] = predicted_class_values[
        "shap_value"
    ].abs()
    top_n = int(config["shap"]["predicted_class_top_contributions"])
    top_contributions = (
        predicted_class_values.sort_values(
            ["display_order", "absolute_shap_value", "gene_name"],
            ascending=[True, False, True],
        )
        .groupby("display_order", as_index=False, group_keys=False)
        .head(top_n)
    )
    top_contributions["absolute_rank"] = top_contributions.groupby("display_order")[
        "absolute_shap_value"
    ].rank(method="first", ascending=False).astype(int)
    top_contributions.to_csv(
        OUTPUT_DIR / "shap_predicted_class_top_contributions.tsv",
        sep="\t",
        index=False,
    )
    if case_summary["absolute_additivity_residual"].max() > 1e-5:
        raise RuntimeError("SHAP additivity residual exceeds 1e-5")
    access.update(
        {
            "status": "COMPLETE",
            "completed_at_utc": now_utc(),
            "locked_test_expression_rows_loaded": len(cases),
            "shap_explanations_generated": len(cases),
            "performance_metrics_computed": 0,
            "model_selection_performed": False,
            "shap_values_sha256": sha256(
                OUTPUT_DIR / "shap_values_selected_cases.tsv.gz"
            ),
        }
    )
    INTERPRETATION_ACCESS_PATH.write_text(
        json.dumps(access, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return [
        "shap_values_selected_cases.tsv.gz",
        "shap_case_summary.tsv",
        "shap_predicted_class_top_contributions.tsv",
    ]


def expected_signal_audit(
    matrix: np.ndarray,
    genes: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict[str, Any],
) -> list[str]:
    markers = list(config["expected_signal_audit"])
    marker_genes = genes.loc[genes["gene_name"].isin(markers)].copy()
    if set(marker_genes["gene_name"]) != set(markers):
        missing = sorted(set(markers) - set(marker_genes["gene_name"]))
        raise RuntimeError(f"Expected-signal audit genes absent from matrix: {missing}")
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    values = np.asarray(
        matrix[
            development["matrix_row"].astype(int).to_numpy()
        ][:, marker_genes["matrix_column"].astype(int).to_numpy()],
        dtype=np.float32,
    )
    expression_records: list[dict[str, Any]] = []
    for subtype in config["class_order"]:
        mask = development["pam50_standard"].eq(subtype).to_numpy()
        for gene_index, gene in marker_genes.reset_index(drop=True).iterrows():
            expression_records.append(
                {
                    "pam50_class": subtype,
                    "gene_name": gene["gene_name"],
                    "matrix_column": int(gene["matrix_column"]),
                    "development_n": int(mask.sum()),
                    "mean_log2_tpm": float(values[mask, gene_index].mean()),
                    "median_log2_tpm": float(np.median(values[mask, gene_index])),
                    "q25_log2_tpm": float(np.quantile(values[mask, gene_index], 0.25)),
                    "q75_log2_tpm": float(np.quantile(values[mask, gene_index], 0.75)),
                }
            )
    expression = pd.DataFrame(expression_records)
    expression.to_csv(
        OUTPUT_DIR / "expected_signal_development_expression.tsv", sep="\t", index=False
    )
    elastic = pd.read_csv(
        OUTPUT_DIR / "elastic_net_coefficient_summary.tsv.gz", sep="\t"
    )
    elastic = elastic.loc[elastic["gene_name"].isin(markers)][
        [
            "pam50_class",
            "matrix_column",
            "gene_name",
            "mean_coefficient_including_zeros",
            "mean_absolute_coefficient_including_zeros",
            "selected_fold_count",
            "nonzero_fold_count",
            "sign_consistency",
            "top20_outer_fold_count",
            "top20_outer_fold_frequency",
        ]
    ]
    permutation = pd.read_csv(
        OUTPUT_DIR / "rf_permutation_importance_summary.tsv.gz", sep="\t"
    )
    permutation = permutation.loc[permutation["gene_name"].isin(markers)][
        [
            "matrix_column",
            "gene_name",
            "selected_fold_count",
            "macro_f1_importance_mean_including_unselected_zeros",
            "balanced_accuracy_importance_mean_including_unselected_zeros",
            "top20_outer_fold_count",
            "top20_outer_fold_frequency",
        ]
    ].rename(
        columns={
            "selected_fold_count": "rf_selected_fold_count",
            "top20_outer_fold_count": "rf_top20_outer_fold_count",
            "top20_outer_fold_frequency": "rf_top20_outer_fold_frequency",
        }
    )
    audit = elastic.merge(
        permutation, on=["matrix_column", "gene_name"], how="left"
    ).sort_values(["gene_name", "pam50_class"])
    audit["audit_status"] = "observed_without_forcing"
    audit.to_csv(OUTPUT_DIR / "expected_signal_audit.tsv", sep="\t", index=False)
    return ["expected_signal_development_expression.tsv", "expected_signal_audit.tsv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError("Interpretability output directory is not empty")
    config = load_config(CONFIG_PATH)
    verify_lock(config)
    evaluation = load_config(EVALUATION_PATH)
    forest = load_config(FOREST_PATH)
    assignments = pd.read_csv(ASSIGNMENT_PATH, sep="\t")
    genes = pd.read_csv(GENE_PATH, sep="\t")
    matrix = np.load(ROOT / evaluation["features"]["matrix"], mmap_mode="r")
    candidate_indices = np.load(
        ROOT / evaluation["features"]["candidate_gene_indices"]
    ).astype(int)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    artifacts = summarize_elastic_net(genes, config)
    if not args.skip_permutation:
        artifacts.extend(
            run_permutation_importance(
                matrix,
                candidate_indices,
                genes,
                assignments,
                evaluation,
                forest,
                config,
            )
        )
    if not args.skip_shap:
        artifacts.extend(
            run_shap(
                matrix,
                candidate_indices,
                genes,
                assignments,
                evaluation,
                forest,
                config,
            )
        )
    if args.skip_permutation or args.skip_shap:
        print("Partial run requested; manifest and expected-signal audit not written", flush=True)
        return
    artifacts.extend(expected_signal_audit(matrix, genes, assignments, config))
    manifest = {
        "schema_version": "1.0.0",
        "status": "COMPLETE",
        "created_at_utc": now_utc(),
        "interpretability_config_sha256": sha256(CONFIG_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "development_n": 756,
        "locked_test_n": 189,
        "permutation_locked_test_expression_rows_loaded": 0,
        "shap_locked_test_expression_rows_loaded": 6,
        "shap_background_partition": "development",
        "shap_background_n": 100,
        "shap_version": shap.__version__,
        "output_sha256": {name: sha256(OUTPUT_DIR / name) for name in artifacts},
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
