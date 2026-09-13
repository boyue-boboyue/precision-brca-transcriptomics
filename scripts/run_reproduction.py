#!/usr/bin/env python3
"""Replay the frozen analysis in an isolated workspace without touching canonical locks."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "metadata",
    "cohort",
    "matrix",
    "eda",
    "train",
    "evaluate",
    "explain",
    "test",
)
PREVIOUS_STAGE = {
    stage: STAGES[index - 1] if index else None
    for index, stage in enumerate(STAGES)
}
GENERATED_CONFIGS = {"final_model_lock_v1.json", "interpretability_v1.json"}
GENERATED_SPLIT_RECORDS = {
    "final_test_access_v1.json",
    "interpretation_test_access_v1.json",
}
LARGE_EXPRESSION_ARRAYS = {
    "counts_uint32.npy",
    "tpm_float32.npy",
    "log2_tpm_float32.npy",
}
ENRICHMENT_RAW_FILES = (
    "gprofiler_data_versions.json",
    "gprofiler_primary_raw.json",
    "gprofiler_consensus_background_raw.json",
)
CANONICAL_EQUIVALENCE_FILES = (
    "outputs/eda/tables/sample_inclusion_flow.tsv",
    "outputs/eda/tables/class_balance.tsv",
    "outputs/eda/tables/analysis_samples.tsv",
    "outputs/modeling/logistic_comparison/outer_fold_metrics.tsv",
    "outputs/modeling/logistic_comparison/outer_fold_predictions.tsv",
    "outputs/modeling/linear_svc/outer_fold_metrics.tsv",
    "outputs/modeling/linear_svc/outer_fold_predictions.tsv",
    "outputs/modeling/random_forest/outer_fold_metrics.tsv",
    "outputs/modeling/random_forest/outer_fold_predictions.tsv",
    "outputs/modeling/pam50_excluded/outer_fold_metrics.tsv",
    "outputs/modeling/pam50_excluded/outer_fold_predictions.tsv",
    "outputs/final_evaluation/selected_hyperparameters.json",
    "outputs/final_evaluation/locked_test_predictions.tsv",
    "outputs/final_evaluation/locked_test_summary.tsv",
    "outputs/final_evaluation/locked_test_per_class_metrics.tsv",
    "outputs/final_evaluation/bootstrap_confidence_intervals.tsv",
    "outputs/performance_report/model_selection.tsv",
    "outputs/performance_report/nested_cv_summary.tsv",
    "outputs/performance_report/oof_per_class_metrics.tsv",
    "outputs/interpretability/elastic_net_top20_stability.tsv",
    "outputs/interpretability/rf_top20_stability.tsv",
    "outputs/interpretability/shap_case_summary.tsv",
    "outputs/interpretability/biological_validation/stable_important_genes.tsv",
    "outputs/interpretability/biological_validation/functional_enrichment_results.tsv",
    "outputs/robustness/scenario_summary.tsv",
    "outputs/robustness/label_source_audit.tsv",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite reproduction record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise RuntimeError(f"Partial reproduction record requires audit: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def source_tree_sha256(base: Path = ROOT) -> str:
    paths = [base / "Makefile", *sorted(base.glob("requirements-*.txt"))]
    for directory in (base / "scripts", base / "tests", base / "config"):
        paths.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    digest = hashlib.sha256()
    for path in paths:
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.parent == base / "config" and path.name in GENERATED_CONFIGS:
            continue
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def canonical_lock_files() -> list[Path]:
    paths = [path for path in (ROOT / "config").glob("*.json") if path.is_file()]
    for directory in (
        ROOT / "data/processed/splits",
        ROOT / "data/processed/interpretability",
    ):
        if directory.exists():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    label_dir = ROOT / "data/processed/labels"
    paths.extend(
        path
        for path in label_dir.glob("*")
        if path.is_file()
        and ("lock" in path.name or path.name == "pam50_signature_genes_v1.tsv")
    )
    return sorted(set(paths))


def canonical_lock_hashes() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in canonical_lock_files()
    }


def assert_canonical_locks_unchanged(expected: dict[str, str]) -> None:
    observed = canonical_lock_hashes()
    if observed != expected:
        changed = sorted(
            path
            for path in set(expected) | set(observed)
            if expected.get(path) != observed.get(path)
        )
        raise RuntimeError(
            "Canonical lock guard failed; changed paths: " + ", ".join(changed)
        )


def resolve_workspace(raw_path: str) -> Path:
    requested = Path(raw_path).expanduser()
    if not requested.is_absolute():
        requested = ROOT / requested
    workspace = requested.resolve(strict=False)
    if workspace == ROOT:
        raise ValueError("The reproduction workspace cannot be the source repository")
    if ROOT in workspace.parents:
        relative = workspace.relative_to(ROOT)
        if relative.parts[:2] != ("work", "reproduction"):
            raise ValueError(
                "A repository-local reproduction workspace must be under work/reproduction/"
            )
    return workspace


def copy_ignore(source: str, names: list[str]) -> list[str]:
    relative = Path(source).resolve().relative_to(ROOT.resolve())
    ignored = {
        name
        for name in names
        if name in {".git", ".venv", "work", "logs", "outputs", "__pycache__", ".DS_Store"}
        or name.endswith((".pyc", ".pyo", ".part"))
    }
    if relative == Path("config"):
        ignored.update(GENERATED_CONFIGS & set(names))
    if relative == Path("data/processed/splits"):
        ignored.update(GENERATED_SPLIT_RECORDS & set(names))
    if relative == Path("data/processed") and "interpretability" in names:
        ignored.add("interpretability")
    if relative == Path("data/processed/expression"):
        ignored.update(LARGE_EXPRESSION_ARRAYS & set(names))
    if relative == Path("data/raw/gdc") and "star_counts" in names:
        ignored.add("star_counts")
    return sorted(ignored)


def environment_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "Pillow",
        "shap",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def prepare_workspace(workspace: Path, lock_hashes: dict[str, str]) -> dict[str, Any]:
    manifest_path = workspace / ".reproduction/manifest.json"
    if workspace.exists():
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Workspace exists without a reproduction manifest; refusing overwrite: {workspace}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_root") != str(ROOT.resolve()):
            raise RuntimeError("Reproduction workspace belongs to another source repository")
        if manifest.get("source_tree_sha256") != source_tree_sha256():
            raise RuntimeError(
                "Source code changed since this workspace was created; choose a new REPRO_DIR"
            )
        if manifest.get("source_tree_sha256") != source_tree_sha256(workspace):
            raise RuntimeError(
                "Reproduction workspace code/configuration changed; choose a new REPRO_DIR"
            )
        if manifest.get("canonical_locks_sha256") != lock_hashes:
            raise RuntimeError(
                "Canonical locks changed since this workspace was created; choose a new REPRO_DIR"
            )
        return manifest

    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT, workspace, ignore=copy_ignore, symlinks=True)
    linked_inputs: dict[str, str] = {}
    for name in sorted(LARGE_EXPRESSION_ARRAYS):
        source = ROOT / "data/processed/expression" / name
        if source.is_file():
            target = workspace / "data/processed/expression" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve())
            linked_inputs[target.relative_to(workspace).as_posix()] = str(source.resolve())
    raw_counts = ROOT / "data/raw/gdc/star_counts"
    if raw_counts.is_dir():
        target = workspace / "data/raw/gdc/star_counts"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(raw_counts.resolve(), target_is_directory=True)
        linked_inputs[target.relative_to(workspace).as_posix()] = str(raw_counts.resolve())

    source_digest = source_tree_sha256()
    if source_tree_sha256(workspace) != source_digest:
        raise RuntimeError("Isolated source snapshot differs from the canonical source tree")

    manifest = {
        "schema_version": "1.0.0",
        "status": "PREPARED",
        "created_at_utc": now_utc(),
        "source_root": str(ROOT.resolve()),
        "source_commit": git_commit(),
        "source_tree_sha256": source_digest,
        "workspace": str(workspace),
        "isolation_policy": (
            "All generated locks, access records, models, reports, and figures remain "
            "inside this workspace; canonical source-repository locks are hash-guarded."
        ),
        "canonical_locks_sha256": lock_hashes,
        "linked_read_only_inputs": linked_inputs,
        "environment": environment_versions(),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def command_plan(
    stage: str,
    *,
    python: str,
    n_jobs: int,
    forest_n_jobs: int,
    download_workers: int,
) -> list[list[str]]:
    commands: dict[str, list[list[str]]] = {
        "metadata": [
            [python, "scripts/verify_source_artifacts.py", "--stage", "metadata"]
        ],
        "cohort": [
            [
                python,
                "scripts/verify_source_artifacts.py",
                "--stage",
                "cohort",
                "--skip-unavailable-publication-supplement",
            ]
        ],
        "matrix": [
            [
                python,
                "scripts/ensure_expression_matrix.py",
                "--download-workers",
                str(download_workers),
            ]
        ],
        "eda": [
            [python, "scripts/run_eda.py"],
            [python, "scripts/verify_eda.py"],
        ],
        "train": [
            [python, "scripts/verify_evaluation_framework.py"],
            [python, "scripts/run_logistic_comparison.py", "--n-jobs", str(n_jobs)],
            [python, "scripts/summarize_logistic_comparison.py"],
            [python, "scripts/verify_logistic_comparison.py"],
            [python, "scripts/run_linear_svc.py", "--n-jobs", str(n_jobs)],
            [python, "scripts/summarize_linear_svc.py"],
            [python, "scripts/verify_linear_svc.py"],
            [
                python,
                "scripts/run_random_forest.py",
                "--forest-n-jobs",
                str(forest_n_jobs),
            ],
            [python, "scripts/summarize_random_forest.py"],
            [python, "scripts/verify_random_forest.py"],
            [
                python,
                "scripts/run_pam50_excluded_nested_cv.py",
                "--n-jobs",
                str(n_jobs),
                "--forest-n-jobs",
                str(forest_n_jobs),
            ],
        ],
        "evaluate": [
            [python, "scripts/lock_final_model.py"],
            [
                python,
                "scripts/run_final_locked_test.py",
                "--forest-n-jobs",
                str(forest_n_jobs),
            ],
            [python, "scripts/generate_unified_performance_report.py"],
            [python, "scripts/verify_unified_performance_report.py"],
        ],
        "explain": [
            [python, "scripts/lock_interpretability_plan.py"],
            [python, "scripts/run_interpretability.py"],
            [python, "scripts/run_biological_validation.py"],
            ["<copy-frozen-enrichment-responses>"],
            [python, "scripts/fetch_functional_enrichment.py"],
            [python, "scripts/generate_interpretability_report.py"],
            [
                python,
                "scripts/run_robustness_analysis.py",
                "--forest-n-jobs",
                str(forest_n_jobs),
            ],
            [python, "scripts/verify_interpretability.py"],
            [python, "scripts/verify_robustness.py"],
        ],
        "test": [
            [python, "-m", "unittest", "discover", "-s", "tests", "-v"],
            [
                python,
                "scripts/verify_source_artifacts.py",
                "--stage",
                "cohort",
                "--skip-unavailable-publication-supplement",
            ],
            [python, "scripts/verify_expression_matrix.py"],
            [python, "scripts/verify_eda.py"],
            [python, "scripts/verify_evaluation_framework.py"],
            [python, "scripts/verify_logistic_comparison.py"],
            [python, "scripts/verify_linear_svc.py"],
            [python, "scripts/verify_random_forest.py"],
            [python, "scripts/verify_unified_performance_report.py"],
            [python, "scripts/verify_interpretability.py"],
            [python, "scripts/verify_robustness.py"],
            ["<compare-key-results-with-canonical-snapshot>"],
        ],
    }
    return commands[stage]


def verifier_plan(stage: str, python: str) -> list[list[str]]:
    plans = {
        "metadata": [[python, "scripts/verify_source_artifacts.py", "--stage", "metadata"]],
        "cohort": [
            [
                python,
                "scripts/verify_source_artifacts.py",
                "--stage",
                "cohort",
                "--skip-unavailable-publication-supplement",
            ]
        ],
        "matrix": [[python, "scripts/verify_expression_matrix.py"]],
        "eda": [[python, "scripts/verify_eda.py"]],
        "train": [
            [python, "scripts/verify_evaluation_framework.py"],
            [python, "scripts/verify_logistic_comparison.py"],
            [python, "scripts/verify_linear_svc.py"],
            [python, "scripts/verify_random_forest.py"],
        ],
        "evaluate": [[python, "scripts/verify_unified_performance_report.py"]],
        "explain": [
            [python, "scripts/verify_interpretability.py"],
            [python, "scripts/verify_robustness.py"],
        ],
        "test": command_plan(
            "test", python=python, n_jobs=1, forest_n_jobs=1, download_workers=1
        )[:-1],
    }
    return plans[stage]


def run_command(command: list[str], workspace: Path) -> None:
    print("+ " + shlex.join(command), flush=True)
    environment = os.environ.copy()
    environment.update(
        {
            "MPLCONFIGDIR": str(workspace / ".matplotlib"),
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    subprocess.run(command, cwd=workspace, env=environment, check=True)


def require_absent(workspace: Path, relatives: tuple[str, ...]) -> None:
    present = [relative for relative in relatives if (workspace / relative).exists()]
    if present:
        raise RuntimeError(
            "Stage has partial or pre-existing outputs and will not overwrite them: "
            + ", ".join(present)
            + ". Use a new REPRO_DIR."
        )


def preflight_stage(stage: str, workspace: Path) -> None:
    guarded = {
        "eda": ("outputs/eda",),
        "train": (
            "outputs/evaluation",
            "outputs/modeling/logistic_comparison",
            "outputs/modeling/linear_svc",
            "outputs/modeling/random_forest",
            "outputs/modeling/pam50_excluded",
        ),
        "evaluate": (
            "config/final_model_lock_v1.json",
            "data/processed/splits/final_test_access_v1.json",
            "outputs/final_evaluation",
            "outputs/performance_report",
        ),
        "explain": (
            "config/interpretability_v1.json",
            "data/processed/interpretability",
            "data/processed/splits/interpretation_test_access_v1.json",
            "outputs/interpretability",
            "outputs/robustness",
        ),
    }
    require_absent(workspace, guarded.get(stage, ()))


def verify_pam50_excluded(workspace: Path) -> None:
    output_dir = workspace / "outputs/modeling/pam50_excluded"
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("Reproduced PAM50-excluded run is incomplete")
    if manifest.get("locked_test_expression_rows_loaded") != 0:
        raise RuntimeError("PAM50-excluded replay loaded locked-test expression")
    if manifest.get("locked_test_predictions_generated") != 0:
        raise RuntimeError("PAM50-excluded replay generated locked-test predictions")
    for relative, expected in manifest["output_sha256"].items():
        if sha256(output_dir / relative) != expected:
            raise RuntimeError(f"PAM50-excluded output hash mismatch: {relative}")

    signature_path = workspace / "data/processed/labels/pam50_signature_genes_v1.tsv"
    with signature_path.open(encoding="utf-8", newline="") as handle:
        signature = {
            int(row["matrix_column"])
            for row in csv.DictReader(handle, delimiter="\t")
        }
    with gzip.open(
        output_dir / "selected_features.tsv.gz", "rt", encoding="utf-8", newline=""
    ) as handle:
        selected = list(csv.DictReader(handle, delimiter="\t"))
    if any(str(row["is_locked_pam50_gene"]).lower() in {"true", "1"} for row in selected):
        raise RuntimeError("PAM50-excluded replay selected a locked PAM50 gene")
    if signature & {int(row["matrix_column"]) for row in selected}:
        raise RuntimeError("PAM50 signature columns leaked into excluded replay")


def comparable_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("created_at_utc", None)
    return value


def seed_enrichment_responses(workspace: Path) -> None:
    relative_dir = Path("outputs/interpretability/biological_validation")
    canonical_dir = ROOT / relative_dir
    reproduction_dir = workspace / relative_dir
    reproduction_request = comparable_request(
        reproduction_dir / "functional_enrichment_request.json"
    )
    canonical_request = comparable_request(
        canonical_dir / "functional_enrichment_request.json"
    )
    if reproduction_request != canonical_request:
        raise RuntimeError(
            "Reproduced enrichment request differs from the frozen canonical request; "
            "refusing to reuse unrelated API responses"
        )
    for name in ENRICHMENT_RAW_FILES:
        source = canonical_dir / name
        target = reproduction_dir / name
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite reproduction enrichment input: {target}")
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError(f"Enrichment response copy failed hash verification: {name}")
    print("+ copied and hash-verified frozen g:Profiler responses", flush=True)


def values_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        left_number = float(left)
        right_number = float(right)
    except ValueError:
        return False
    if math.isnan(left_number) and math.isnan(right_number):
        return True
    return math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-10)


def tabular_equivalent(canonical: Path, reproduced: Path) -> bool:
    with canonical.open(encoding="utf-8", newline="") as left_handle:
        left = list(csv.reader(left_handle, delimiter="\t"))
    with reproduced.open(encoding="utf-8", newline="") as right_handle:
        right = list(csv.reader(right_handle, delimiter="\t"))
    if len(left) != len(right):
        return False
    return all(
        len(left_row) == len(right_row)
        and all(
            values_equivalent(left_value, right_value)
            for left_value, right_value in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left, right, strict=True)
    )


def json_equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            json_equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_equivalent(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-10)
    return left == right


def scientific_equivalence(canonical: Path, reproduced: Path) -> tuple[bool, str]:
    if sha256(canonical) == sha256(reproduced):
        return True, "exact_sha256"
    if canonical.suffix == ".tsv" and tabular_equivalent(canonical, reproduced):
        return True, "numeric_tolerance_1e-9"
    if canonical.suffix == ".json":
        left = json.loads(canonical.read_text(encoding="utf-8"))
        right = json.loads(reproduced.read_text(encoding="utf-8"))
        if json_equivalent(left, right):
            return True, "numeric_tolerance_1e-9"
    return False, "mismatch"


def compare_with_canonical(workspace: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    exact_match_count = 0
    tolerance_match_count = 0
    for relative in CANONICAL_EQUIVALENCE_FILES:
        canonical = ROOT / relative
        reproduced = workspace / relative
        canonical_hash = sha256(canonical) if canonical.is_file() else None
        reproduced_hash = sha256(reproduced) if reproduced.is_file() else None
        if canonical_hash is not None and reproduced_hash is not None:
            matches, comparison = scientific_equivalence(canonical, reproduced)
        else:
            matches, comparison = False, "missing"
        exact_match_count += int(comparison == "exact_sha256")
        tolerance_match_count += int(comparison == "numeric_tolerance_1e-9")
        records[relative] = {
            "canonical_sha256": canonical_hash,
            "reproduced_sha256": reproduced_hash,
            "matches": matches,
            "comparison": comparison,
        }
        if not matches:
            mismatches.append(relative)
    report = {
        "status": "PASS" if not mismatches else "FAIL",
        "checked_at_utc": now_utc(),
        "exact_match_count": exact_match_count,
        "numeric_tolerance_match_count": tolerance_match_count,
        "checked_file_count": len(records),
        "mismatches": mismatches,
        "files": records,
    }
    path = workspace / ".reproduction/canonical_equivalence.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if mismatches:
        raise RuntimeError(
            "Reproduction differs from canonical key results: " + ", ".join(mismatches)
        )
    return report


def execute_stage(
    stage: str,
    workspace: Path,
    commands: list[list[str]],
    lock_hashes: dict[str, str],
) -> None:
    marker = workspace / ".reproduction/stages" / f"{stage}.json"
    # Keep the virtual-environment launcher path. Resolving its symlink would
    # bypass the venv prefix and lose packages installed only in that environment.
    python = sys.executable
    if marker.exists():
        status = json.loads(marker.read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"Incomplete reproduction stage marker: {marker}")
        print(f"[{stage}] already complete; re-verifying without overwriting locks", flush=True)
        for command in verifier_plan(stage, python):
            run_command(command, workspace)
        if stage == "train":
            verify_pam50_excluded(workspace)
        if stage == "test":
            compare_with_canonical(workspace)
        assert_canonical_locks_unchanged(lock_hashes)
        return

    previous = PREVIOUS_STAGE[stage]
    if previous is not None:
        previous_marker = workspace / ".reproduction/stages" / f"{previous}.json"
        if not previous_marker.is_file():
            raise RuntimeError(f"Stage {stage} requires completed stage {previous}")
    preflight_stage(stage, workspace)
    started = now_utc()
    for command in commands:
        if command == ["<copy-frozen-enrichment-responses>"]:
            seed_enrichment_responses(workspace)
        elif command == ["<compare-key-results-with-canonical-snapshot>"]:
            compare_with_canonical(workspace)
        else:
            run_command(command, workspace)
    if stage == "train":
        verify_pam50_excluded(workspace)
    assert_canonical_locks_unchanged(lock_hashes)
    atomic_json(
        marker,
        {
            "schema_version": "1.0.0",
            "status": "COMPLETE",
            "stage": stage,
            "started_at_utc": started,
            "completed_at_utc": now_utc(),
            "commands": [shlex.join(command) for command in commands],
            "canonical_locks_unchanged": True,
            "canonical_locks_sha256": lock_hashes,
        },
    )
    print(f"[{stage}] reproduction stage complete: {marker}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--workspace", default="work/reproduction/default")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--forest-n-jobs", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_jobs == 0 or args.forest_n_jobs == 0 or args.download_workers < 1:
        raise ValueError("Job counts cannot be zero and download-workers must be positive")
    workspace = resolve_workspace(args.workspace)
    python = sys.executable
    commands = command_plan(
        args.stage,
        python=python,
        n_jobs=args.n_jobs,
        forest_n_jobs=args.forest_n_jobs,
        download_workers=args.download_workers,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "DRY_RUN",
                    "stage": args.stage,
                    "workspace": str(workspace),
                    "previous_stage": PREVIOUS_STAGE[args.stage],
                    "commands": [shlex.join(command) for command in commands],
                    "canonical_repository_writes": 0,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    lock_hashes = canonical_lock_hashes()
    prepare_workspace(workspace, lock_hashes)
    try:
        execute_stage(args.stage, workspace, commands, lock_hashes)
    finally:
        assert_canonical_locks_unchanged(lock_hashes)


if __name__ == "__main__":
    main()
