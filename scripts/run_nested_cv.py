#!/usr/bin/env python3
"""Run development-only nested CV using the precomputed locked assignments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

try:
    from evaluation_framework import build_pipeline, inner_splits, load_config, parameter_grid
except ModuleNotFoundError:  # pragma: no cover - package import
    from scripts.evaluation_framework import (
        build_pipeline,
        inner_splits,
        load_config,
        parameter_grid,
    )


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["four_class", "five_class"], default="four_class")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy_prior", "multinomial_logistic", "linear_svm", "random_forest"],
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "modeling" / "development_nested_cv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ROOT / "config" / "evaluation.json"
    config = load_config(config_path)
    assignment_path = (
        ROOT
        / "data"
        / "processed"
        / "splits"
        / f"{args.cohort}_split_assignments.tsv"
    )
    assignments = pd.read_csv(assignment_path, sep="\t")
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    if development["outer_fold"].isna().any():
        raise RuntimeError("Development patients must have an outer fold")
    if assignments.loc[
        assignments["holdout_split"].eq("locked_test"), "outer_fold"
    ].notna().any():
        raise RuntimeError("Locked-test patients must not have CV assignments")

    matrix = np.load(ROOT / config["features"]["matrix"], mmap_mode="r")
    coding_indices = np.load(ROOT / config["features"]["candidate_gene_indices"])
    rows = development["matrix_row"].astype(int).to_numpy()
    X = np.asarray(matrix[rows][:, coding_indices], dtype=np.float32)
    y = development[config["label_column"]].astype(str).to_numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    parameter_records: list[dict[str, object]] = []
    outer_values = development["outer_fold"].astype(int).to_numpy()
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}

    for model_name in args.models:
        if model_name not in config["models"]:
            raise ValueError(f"Unknown model: {model_name}")
        for outer_fold in sorted(np.unique(outer_values)):
            outer_valid_index = np.flatnonzero(outer_values == outer_fold)
            outer_train_index = np.flatnonzero(outer_values != outer_fold)
            cv = inner_splits(development, int(outer_fold))
            pipeline = build_pipeline(
                model_name,
                config,
                random_seed=int(config["random_seed"]) + 10000 + int(outer_fold),
            )
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=parameter_grid(model_name, config),
                scoring=scoring,
                refit=config["selection"]["inner_refit_metric"],
                cv=cv,
                n_jobs=args.n_jobs,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(X[outer_train_index], y[outer_train_index])
            predicted = search.predict(X[outer_valid_index])
            truth = y[outer_valid_index]
            metrics_records.append(
                {
                    "cohort": args.cohort,
                    "model": model_name,
                    "outer_fold": int(outer_fold),
                    "outer_train_n": int(len(outer_train_index)),
                    "outer_validation_n": int(len(outer_valid_index)),
                    "macro_f1": float(f1_score(truth, predicted, average="macro")),
                    "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
                    "accuracy": float(accuracy_score(truth, predicted)),
                    "inner_best_macro_f1": float(search.best_score_),
                }
            )
            parameter_records.append(
                {
                    "cohort": args.cohort,
                    "model": model_name,
                    "outer_fold": int(outer_fold),
                    "best_params_json": json.dumps(search.best_params_, sort_keys=True),
                }
            )
            for local_index, observed, estimate in zip(
                outer_valid_index, truth, predicted, strict=True
            ):
                row = development.iloc[int(local_index)]
                prediction_records.append(
                    {
                        "cohort": args.cohort,
                        "model": model_name,
                        "outer_fold": int(outer_fold),
                        "matrix_row": int(row["matrix_row"]),
                        "case_barcode": row["case_barcode"],
                        "sample_barcode": row["sample_barcode"],
                        "observed": observed,
                        "predicted": estimate,
                    }
                )

    pd.DataFrame(metrics_records).to_csv(
        args.output_dir / f"{args.cohort}_outer_fold_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(prediction_records).to_csv(
        args.output_dir / f"{args.cohort}_outer_fold_predictions.tsv", sep="\t", index=False
    )
    pd.DataFrame(parameter_records).to_csv(
        args.output_dir / f"{args.cohort}_best_parameters.tsv", sep="\t", index=False
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort": args.cohort,
        "models": args.models,
        "development_n": int(len(development)),
        "locked_test_rows_loaded_for_modeling": 0,
        "random_seed": int(config["random_seed"]),
        "selection_metric": config["selection"]["inner_refit_metric"],
    }
    with (args.output_dir / f"{args.cohort}_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
