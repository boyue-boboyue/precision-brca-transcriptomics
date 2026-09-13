from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import run_reproduction


class ReproductionModeTests(unittest.TestCase):
    def test_stage_order_is_complete_and_linear(self) -> None:
        self.assertEqual(
            run_reproduction.STAGES,
            (
                "metadata",
                "cohort",
                "matrix",
                "eda",
                "train",
                "evaluate",
                "explain",
                "test",
            ),
        )
        self.assertIsNone(run_reproduction.PREVIOUS_STAGE["metadata"])
        self.assertEqual(run_reproduction.PREVIOUS_STAGE["test"], "explain")

    def test_repository_local_workspace_is_safely_scoped(self) -> None:
        with self.assertRaises(ValueError):
            run_reproduction.resolve_workspace(str(run_reproduction.ROOT))
        with self.assertRaises(ValueError):
            run_reproduction.resolve_workspace("outputs/reproduction")
        expected = run_reproduction.ROOT / "work/reproduction/unit-test"
        self.assertEqual(
            run_reproduction.resolve_workspace("work/reproduction/unit-test"),
            expected,
        )

    def test_copy_policy_omits_canonical_generated_locks_and_large_arrays(self) -> None:
        config_ignored = run_reproduction.copy_ignore(
            str(run_reproduction.ROOT / "config"),
            ["evaluation.json", "final_model_lock_v1.json", "interpretability_v1.json"],
        )
        self.assertEqual(
            set(config_ignored),
            {"final_model_lock_v1.json", "interpretability_v1.json"},
        )
        array_ignored = run_reproduction.copy_ignore(
            str(run_reproduction.ROOT / "data/processed/expression"),
            ["genes.tsv", "counts_uint32.npy", "log2_tpm_float32.npy"],
        )
        self.assertEqual(
            set(array_ignored),
            {"counts_uint32.npy", "log2_tpm_float32.npy"},
        )

    def test_full_plan_creates_new_locks_only_inside_reproduction(self) -> None:
        python = str(Path(".venv/bin/python"))
        evaluation = run_reproduction.command_plan(
            "evaluate",
            python=python,
            n_jobs=1,
            forest_n_jobs=2,
            download_workers=3,
        )
        explanation = run_reproduction.command_plan(
            "explain",
            python=python,
            n_jobs=1,
            forest_n_jobs=2,
            download_workers=3,
        )
        self.assertIn([python, "scripts/lock_final_model.py"], evaluation)
        self.assertIn([python, "scripts/lock_interpretability_plan.py"], explanation)
        self.assertIn(["<copy-frozen-enrichment-responses>"], explanation)

    def test_equivalence_allows_only_tiny_numeric_serialization_differences(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical.tsv"
            reproduced = root / "reproduced.tsv"
            canonical.write_text("case\tmetric\nA\t0.9000000000\n", encoding="utf-8")
            reproduced.write_text("case\tmetric\nA\t0.9000000001\n", encoding="utf-8")
            self.assertEqual(
                run_reproduction.scientific_equivalence(canonical, reproduced),
                (True, "numeric_tolerance_1e-9"),
            )
            reproduced.write_text("case\tmetric\nB\t0.9000000001\n", encoding="utf-8")
            self.assertEqual(
                run_reproduction.scientific_equivalence(canonical, reproduced),
                (False, "mismatch"),
            )

    def test_stage_preflight_refuses_to_overwrite_generated_lock(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            lock = workspace / "config/final_model_lock_v1.json"
            lock.parent.mkdir(parents=True)
            lock.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                run_reproduction.preflight_stage("evaluate", workspace)

    def test_audit_record_is_append_only(self) -> None:
        with TemporaryDirectory() as directory:
            record = Path(directory) / "stage.json"
            run_reproduction.atomic_json(record, {"status": "COMPLETE"})
            with self.assertRaises(RuntimeError):
                run_reproduction.atomic_json(record, {"status": "REPLACED"})
            self.assertEqual(record.read_text(encoding="utf-8"), '{\n  "status": "COMPLETE"\n}\n')


if __name__ == "__main__":
    unittest.main()
