from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from sklearn.base import clone

from scripts.evaluation_framework import (
    LowExpressionFilter,
    TopVarianceSelector,
    build_pipeline,
)
from scripts.lock_evaluation_splits import attach_nested_folds, stratified_holdout_groups


ROOT = Path(__file__).resolve().parents[1]


class LeakageSafeTransformerTests(unittest.TestCase):
    def test_low_expression_support_is_learned_from_training_only(self) -> None:
        training = np.array(
            [
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 1.5],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.5],
            ],
            dtype=np.float32,
        )
        validation = np.array([[0.0, 9.0, 0.0]], dtype=np.float32)
        transformer = LowExpressionFilter(threshold=1.0, min_fraction=0.5).fit(training)
        np.testing.assert_array_equal(transformer.get_support(), [True, False, True])
        transformed = transformer.transform(validation)
        self.assertEqual(transformed.shape, (1, 2))
        np.testing.assert_array_equal(transformer.get_support(), [True, False, True])

    def test_top_variance_support_is_learned_from_training_only(self) -> None:
        training = np.array([[0, 0, 0], [1, 3, 2], [2, 6, 4]], dtype=np.float32)
        validation = np.array([[100, 0, 0]], dtype=np.float32)
        transformer = TopVarianceSelector(k=1).fit(training)
        np.testing.assert_array_equal(transformer.get_support(indices=True), [1])
        self.assertEqual(transformer.transform(validation).item(), 0.0)

    def test_pipeline_order_and_cloneability(self) -> None:
        config = json.loads((ROOT / "config" / "evaluation.json").read_text())
        pipeline = build_pipeline("multinomial_logistic", config)
        self.assertEqual(
            list(pipeline.named_steps),
            [
                "low_expression_filter",
                "median_imputer",
                "variance_selector",
                "scaler",
                "classifier",
            ],
        )
        clone(pipeline)

    def test_pipeline_fits_with_missing_values(self) -> None:
        config = json.loads((ROOT / "config" / "evaluation.json").read_text())
        rng = np.random.default_rng(17)
        X = rng.normal(loc=2.0, scale=1.0, size=(60, 12)).astype(np.float32)
        X[0, 0] = np.nan
        y = np.repeat(["A", "B", "C"], 20)
        pipeline = build_pipeline("multinomial_logistic", config)
        pipeline.set_params(variance_selector__k=5, classifier__C=0.1)
        pipeline.fit(X, y)
        self.assertEqual(pipeline.predict(X[:4]).shape, (4,))

    def test_duplicate_samples_remain_in_the_same_patient_folds(self) -> None:
        records = []
        for class_index, label in enumerate(["A", "B", "C"]):
            for patient_index in range(15):
                patient = f"P{class_index}-{patient_index:02d}"
                for sample_index in range(2):
                    records.append(
                        {
                            "matrix_row": len(records),
                            "case_id": patient,
                            "case_barcode": patient,
                            "sample_id": f"{patient}-S{sample_index}",
                            "sample_barcode": f"{patient}-S{sample_index}",
                            "pam50_standard": label,
                        }
                    )
        import pandas as pd

        rows = pd.DataFrame(records)
        holdout = stratified_holdout_groups(
            rows,
            group_column="case_barcode",
            label_column="pam50_standard",
            test_fraction=0.2,
            seed=11,
        )
        assigned = attach_nested_folds(
            rows,
            holdout,
            group_column="case_barcode",
            label_column="pam50_standard",
            outer_folds=5,
            inner_folds=5,
            seed=11,
        )
        fold_columns = [
            "holdout_split",
            "outer_fold",
            "inner_fold_outer_1",
            "inner_fold_outer_2",
            "inner_fold_outer_3",
            "inner_fold_outer_4",
            "inner_fold_outer_5",
        ]
        for column in fold_columns:
            self.assertTrue(
                assigned.groupby("patient_group", dropna=False)[column]
                .nunique(dropna=False)
                .eq(1)
                .all()
            )


if __name__ == "__main__":
    unittest.main()
