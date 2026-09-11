#!/usr/bin/env python3
"""Leakage-safe preprocessing pipelines and deterministic nested-CV helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class LowExpressionFilter(BaseEstimator, TransformerMixin):
    """Keep genes expressed above a threshold in a training-fold fraction.

    For log2(TPM + 1), ``threshold=1`` is equivalent to TPM >= 1.  The learned
    mask is immutable during ``transform`` and therefore cannot use validation
    or locked-test distributions.
    """

    def __init__(self, threshold: float = 1.0, min_fraction: float = 0.10):
        self.threshold = threshold
        self.min_fraction = min_fraction

    def fit(self, X: Any, y: Any = None) -> "LowExpressionFilter":
        values = _as_2d_float_array(X)
        if not 0 < self.min_fraction <= 1:
            raise ValueError("min_fraction must be in (0, 1]")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        self.n_features_in_ = values.shape[1]
        finite = np.isfinite(values)
        expressed = finite & (values >= self.threshold)
        self.expression_fraction_ = expressed.mean(axis=0)
        self.support_ = self.expression_fraction_ >= self.min_fraction
        if not self.support_.any():
            raise ValueError("Low-expression filter removed every feature")
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = _as_2d_float_array(X)
        _check_feature_count(values, self.n_features_in_)
        return values[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        if not hasattr(self, "support_"):
            raise AttributeError("LowExpressionFilter is not fitted")
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = _feature_names(input_features, self.n_features_in_)
        return names[self.support_]


class TopVarianceSelector(BaseEstimator, TransformerMixin):
    """Select the highest-variance features using training-fold data only."""

    def __init__(self, k: int | str = 1000):
        self.k = k

    def fit(self, X: Any, y: Any = None) -> "TopVarianceSelector":
        values = _as_2d_float_array(X)
        self.n_features_in_ = values.shape[1]
        if self.k == "all":
            selected_count = self.n_features_in_
        elif isinstance(self.k, (int, np.integer)) and self.k > 0:
            selected_count = min(int(self.k), self.n_features_in_)
        else:
            raise ValueError("k must be a positive integer or 'all'")
        self.variances_ = np.var(values, axis=0)
        if not np.isfinite(self.variances_).all():
            raise ValueError("TopVarianceSelector requires finite imputed input")
        ranked = np.argsort(-self.variances_, kind="stable")
        self.selected_indices_ = np.sort(ranked[:selected_count])
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_indices_] = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = _as_2d_float_array(X)
        _check_feature_count(values, self.n_features_in_)
        return values[:, self.selected_indices_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        if not hasattr(self, "support_"):
            raise AttributeError("TopVarianceSelector is not fitted")
        return self.selected_indices_.copy() if indices else self.support_.copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = _feature_names(input_features, self.n_features_in_)
        return names[self.selected_indices_]


def _as_2d_float_array(X: Any) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, received shape {values.shape}")
    if np.isinf(values).any():
        raise ValueError("Expression matrix contains infinite values")
    return values


def _check_feature_count(values: np.ndarray, expected: int) -> None:
    if values.shape[1] != expected:
        raise ValueError(
            f"Feature count changed between fit and transform: "
            f"expected {expected}, received {values.shape[1]}"
        )


def _feature_names(input_features: Any, n_features: int) -> np.ndarray:
    if input_features is None:
        return np.asarray([f"x{i}" for i in range(n_features)], dtype=object)
    names = np.asarray(input_features, dtype=object)
    if len(names) != n_features:
        raise ValueError("input_features length does not match fitted feature count")
    return names


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def build_pipeline(
    model_name: str,
    config: dict[str, Any],
    *,
    random_seed: int | None = None,
) -> Pipeline:
    """Build the only permitted preprocessing-to-classifier model path."""

    seed = int(config["random_seed"] if random_seed is None else random_seed)
    feature_config = config["features"]
    low_config = feature_config["low_expression_filter"]
    classifier = _build_classifier(model_name, config, seed)
    return Pipeline(
        steps=[
            (
                "low_expression_filter",
                LowExpressionFilter(
                    threshold=float(low_config["log2_tpm_threshold"]),
                    min_fraction=float(low_config["minimum_training_fraction"]),
                ),
            ),
            ("median_imputer", SimpleImputer(strategy="median")),
            ("variance_selector", TopVarianceSelector(k=1000)),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def parameter_grid(model_name: str, config: dict[str, Any]) -> dict[str, list[Any]]:
    try:
        grid = config["models"][model_name]["grid"]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}") from exc
    return {key: list(value) for key, value in grid.items()}


def _build_classifier(
    model_name: str, config: dict[str, Any], seed: int
) -> BaseEstimator:
    if model_name not in config["models"]:
        raise ValueError(f"Unknown model {model_name!r}")
    fixed = dict(config["models"][model_name]["fixed"])
    if model_name == "dummy_prior":
        return DummyClassifier(**fixed, random_state=seed)
    if model_name == "multinomial_logistic":
        return LogisticRegression(
            **fixed,
            random_state=seed,
        )
    if model_name == "linear_svm":
        return LinearSVC(
            **fixed,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            **fixed,
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"No classifier builder registered for {model_name!r}")


def outer_splits(assignments: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield development-only outer train/validation indices."""

    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    folds = sorted(development["outer_fold"].astype(int).unique())
    for fold in folds:
        valid = np.flatnonzero(development["outer_fold"].to_numpy() == fold)
        train = np.flatnonzero(development["outer_fold"].to_numpy() != fold)
        yield train, valid


def inner_splits(
    development: pd.DataFrame, outer_fold: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return inner indices relative to the requested outer-training subset."""

    outer_train = development.loc[
        development["outer_fold"].astype(int).ne(int(outer_fold))
    ].reset_index(drop=True)
    column = f"inner_fold_outer_{int(outer_fold)}"
    if column not in outer_train:
        raise ValueError(f"Missing precomputed assignment column {column}")
    values = outer_train[column].astype(int).to_numpy()
    return [
        (np.flatnonzero(values != fold), np.flatnonzero(values == fold))
        for fold in sorted(np.unique(values))
    ]
