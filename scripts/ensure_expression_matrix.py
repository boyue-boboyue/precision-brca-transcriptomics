#!/usr/bin/env python3
"""Verify expression arrays or reconstruct Git-omitted arrays from frozen sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPRESSION_DIR = ROOT / "data/processed/expression"
MANIFEST_PATH = EXPRESSION_DIR / "matrix_manifest.json"
GDC_MANIFEST_PATH = ROOT / "data/manifests/gdc_manifest_tcga_brca_star_counts.tsv"
GDC_DOWNLOAD_ROOT = ROOT / "data/raw/gdc/star_counts"
GDC_VERIFICATION_TABLE = ROOT / "data/metadata/gdc_download_verification.tsv"
GDC_VERIFICATION_SUMMARY = (
    ROOT / "data/metadata/gdc_download_verification_summary.json"
)
LARGE_ARRAYS = (
    EXPRESSION_DIR / "counts_uint32.npy",
    EXPRESSION_DIR / "tpm_float32.npy",
    EXPRESSION_DIR / "log2_tpm_float32.npy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_gdc_download() -> None:
    with GDC_MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    rows: list[dict[str, object]] = []
    expected_bytes = 0
    actual_bytes = 0
    verified_count = 0
    for record in records:
        target = GDC_DOWNLOAD_ROOT / record["id"] / record["filename"]
        expected_size = int(record["size"])
        expected_bytes += expected_size
        actual_size = target.stat().st_size if target.is_file() else None
        actual_md5 = md5(target) if target.is_file() else ""
        if actual_size is not None:
            actual_bytes += actual_size
        status = (
            "OK"
            if actual_size == expected_size and actual_md5 == record["md5"]
            else ("MISSING" if actual_size is None else "FAILED")
        )
        verified_count += int(status == "OK")
        rows.append(
            {
                "file_id": record["id"],
                "file_name": record["filename"],
                "expected_size": expected_size,
                "actual_size": "" if actual_size is None else actual_size,
                "expected_md5": record["md5"],
                "actual_md5": actual_md5,
                "status": status,
            }
        )

    GDC_VERIFICATION_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with GDC_VERIFICATION_TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    failed_count = len(rows) - verified_count
    summary = {
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expected_file_count": len(rows),
        "verified_file_count": verified_count,
        "failed_file_count": failed_count,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "expected_gib": expected_bytes / 1073741824,
        "actual_gib": actual_bytes / 1073741824,
        "all_files_verified": (
            failed_count == 0
            and expected_bytes == actual_bytes
            and len(rows) == verified_count
        ),
    }
    GDC_VERIFICATION_SUMMARY.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["all_files_verified"]:
        raise RuntimeError("One or more locked GDC downloads failed verification")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_workers < 1:
        raise ValueError("download-workers must be at least 1")

    present = [path.is_file() for path in LARGE_ARRAYS]
    if all(present):
        run([sys.executable, "scripts/verify_expression_matrix.py"])
        return

    missing = [str(path.relative_to(ROOT)) for path in LARGE_ARRAYS if not path.is_file()]
    print(
        "Expression arrays are incomplete; reconstructing from the locked GDC manifest. "
        f"Missing: {', '.join(missing)}",
        flush=True,
    )
    canonical_bytes = MANIFEST_PATH.read_bytes() if MANIFEST_PATH.is_file() else None
    canonical = json.loads(canonical_bytes) if canonical_bytes is not None else None
    if canonical is not None:
        builder_path = ROOT / "scripts/build_expression_matrix.py"
        if sha256(builder_path) != canonical.get("builder_script_sha256"):
            raise RuntimeError(
                "Expression builder differs from the version frozen in the matrix manifest"
            )
    environment = os.environ.copy()
    environment["GDC_DOWNLOAD_WORKERS"] = str(args.download_workers)

    run(["bash", "scripts/download_gdc_star_counts.sh"], env=environment)
    verify_gdc_download()
    try:
        run([sys.executable, "scripts/build_expression_matrix.py"])
        if canonical is not None:
            rebuilt = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if rebuilt.get("artifacts") != canonical.get("artifacts"):
                raise RuntimeError(
                    "Rebuilt matrix artifact hashes differ from the frozen manifest"
                )
            # The build timestamp is intentionally volatile. Restore the committed
            # manifest only after every reconstructed artifact matches it exactly.
            MANIFEST_PATH.write_bytes(canonical_bytes)
    except BaseException:
        if canonical_bytes is not None:
            MANIFEST_PATH.write_bytes(canonical_bytes)
        raise

    run([sys.executable, "scripts/verify_expression_matrix.py"])


if __name__ == "__main__":
    main()
