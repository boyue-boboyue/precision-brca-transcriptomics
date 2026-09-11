#!/usr/bin/env python3
"""Verify handoff archive checksums, member names, and member sizes."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF_DIR = ROOT / "outputs" / "handoff"
EXPORT_DIR = ROOT / "outputs" / "exports"
PREFIX = "precision-brca-transcriptomics"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    summary = json.loads((HANDOFF_DIR / "export_summary.json").read_text())
    inventory = json.loads((HANDOFF_DIR / "export_inventory.json").read_text())
    results = []
    inventory_keys = {"core": "core_archive", "processed_data": "processed_data_archive"}
    for archive_record in summary["archives"]:
        archive = EXPORT_DIR / archive_record["file_name"]
        require(archive.exists(), f"Missing archive: {archive}")
        require(archive.stat().st_size == archive_record["size_bytes"], f"Size mismatch: {archive}")
        require(sha256(archive) == archive_record["sha256"], f"SHA-256 mismatch: {archive}")
        expected_files = inventory[inventory_keys[archive_record["kind"]]]["files"]
        expected = {
            str(Path(PREFIX) / record["path"]): record["size_bytes"]
            for record in expected_files
        }
        inventory_member = str(Path(PREFIX) / "outputs" / "handoff" / "export_inventory.json")
        expected[inventory_member] = (HANDOFF_DIR / "export_inventory.json").stat().st_size
        with tarfile.open(archive, mode="r:gz") as tar:
            observed = {member.name: member.size for member in tar.getmembers() if member.isfile()}
        require(observed == expected, f"Archive member mismatch: {archive}")
        require(not any("data/raw/gdc/star_counts" in name for name in observed), "Raw STAR Counts entered an archive")
        require(not any("/.venv/" in name for name in observed), "Virtual environment entered an archive")
        results.append(
            {
                "file_name": archive.name,
                "status": "PASS",
                "size_bytes": archive.stat().st_size,
                "sha256": archive_record["sha256"],
                "member_count": len(observed),
            }
        )
    report = {
        "status": "PASS",
        "archives": results,
        "raw_gdc_star_counts_omitted": True,
        "virtual_environment_omitted": True,
    }
    (HANDOFF_DIR / "export_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
