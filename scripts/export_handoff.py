#!/usr/bin/env python3
"""Create portable core and processed-data handoff archives."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF_DIR = ROOT / "outputs" / "handoff"
EXPORT_DIR = ROOT / "outputs" / "exports"
ARCHIVE_PREFIX = "precision-brca-transcriptomics"
SNAPSHOT_DATE = "2026-09-09"
CORE_ARCHIVE = EXPORT_DIR / f"oncostratify-brca-core-{SNAPSHOT_DATE}.tar.gz"
DATA_ARCHIVE = EXPORT_DIR / f"oncostratify-brca-processed-data-{SNAPSHOT_DATE}.tar.gz"
LARGE_MATRICES = {
    "counts_uint32.npy",
    "tpm_float32.npy",
    "log2_tpm_float32.npy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_tree(path: Path, *, exclude_names: set[str] | None = None) -> list[Path]:
    exclude_names = exclude_names or set()
    if not path.exists():
        return []
    if path.is_file():
        return [] if path.name in exclude_names else [path]
    return [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file()
        and item.name not in exclude_names
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    ]


def unique(paths: list[Path]) -> list[Path]:
    return sorted(set(paths), key=lambda path: str(path.relative_to(ROOT)))


def record_files(paths: list[Path]) -> list[dict]:
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def create_archive(archive: Path, paths: list[Path], inventory_path: Path) -> None:
    with tarfile.open(archive, mode="w:gz", compresslevel=1) as tar:
        for path in paths + [inventory_path]:
            relative = path.relative_to(ROOT)
            tar.add(
                path,
                arcname=str(Path(ARCHIVE_PREFIX) / relative),
                recursive=False,
                filter=tar_filter,
            )


def main() -> None:
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "docs" / "handoff.md", HANDOFF_DIR / "README.md")

    common_root_files = [
        ROOT / ".gitignore",
        ROOT / "README.md",
        ROOT / "requirements-expression.txt",
        ROOT / "requirements-eda.txt",
    ]
    core_paths = common_root_files.copy()
    for location in [
        ROOT / "docs",
        ROOT / "scripts",
        ROOT / "data" / "manifests",
        ROOT / "data" / "metadata",
        ROOT / "data" / "processed" / "labels",
        ROOT / "data" / "raw" / "pancanatlas",
        ROOT / "data" / "raw" / "cbioportal",
        ROOT / "data" / "raw" / "gdc" / "publication_supplement",
        ROOT / "outputs" / "eda",
    ]:
        core_paths.extend(collect_tree(location))
    core_paths.extend(collect_tree(ROOT / "data" / "processed" / "expression", exclude_names=LARGE_MATRICES))
    core_paths.extend([ROOT / "data" / "README.md", HANDOFF_DIR / "README.md"])
    core_paths = unique([path for path in core_paths if path.exists()])

    data_paths = common_root_files.copy()
    for location in [
        ROOT / "docs",
        ROOT / "scripts",
        ROOT / "data" / "processed" / "expression",
        ROOT / "data" / "processed" / "labels",
    ]:
        data_paths.extend(collect_tree(location))
    data_paths.extend([ROOT / "data" / "README.md", HANDOFF_DIR / "README.md"])
    data_paths = unique([path for path in data_paths if path.exists()])

    raw_star_dir = ROOT / "data" / "raw" / "gdc" / "star_counts"
    raw_star_files = collect_tree(raw_star_dir)
    inventory = {
        "snapshot_date": SNAPSHOT_DATE,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_archive_prefix": ARCHIVE_PREFIX,
        "current_state": {
            "expression_matrix_status": "PASS",
            "eda_status": "PASS",
            "matrix_samples": 1095,
            "matrix_genes": 60660,
            "pam50_five_class": 981,
            "pam50_four_class": 945,
            "supervised_modelling_started": False,
        },
        "core_archive": {
            "file_name": CORE_ARCHIVE.name,
            "description": "Code, protocol, metadata, locked labels, matrix axes/indices, and EDA; excludes the three large expression arrays.",
            "files": record_files(core_paths),
        },
        "processed_data_archive": {
            "file_name": DATA_ARCHIVE.name,
            "description": "Processed expression arrays, axes, labels, verification files, protocol, and scripts required to continue modelling.",
            "files": record_files(data_paths),
        },
        "intentionally_omitted": {
            "raw_gdc_star_counts": {
                "path": str(raw_star_dir.relative_to(ROOT)),
                "file_count": len(raw_star_files),
                "size_bytes": sum(path.stat().st_size for path in raw_star_files),
                "reason": "Already represented by verified processed matrices; can be re-downloaded from the saved GDC manifest.",
            },
            "virtual_environment": {
                "path": ".venv",
                "reason": "Platform-specific and reproducible from pinned requirement files.",
            },
        },
    }
    inventory_path = HANDOFF_DIR / "export_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")

    create_archive(CORE_ARCHIVE, core_paths, inventory_path)
    create_archive(DATA_ARCHIVE, data_paths, inventory_path)

    archive_records = []
    for archive, kind, file_count in [
        (CORE_ARCHIVE, "core", len(core_paths) + 1),
        (DATA_ARCHIVE, "processed_data", len(data_paths) + 1),
    ]:
        archive_records.append(
            {
                "kind": kind,
                "file_name": archive.name,
                "size_bytes": archive.stat().st_size,
                "sha256": sha256(archive),
                "member_count": file_count,
            }
        )

    summary = {
        "snapshot_date": SNAPSHOT_DATE,
        "generated_at_utc": inventory["generated_at_utc"],
        "archives": archive_records,
        "restore": [
            f"tar -xzf {CORE_ARCHIVE.name}",
            f"tar -xzf {DATA_ARCHIVE.name}",
            f"cd {ARCHIVE_PREFIX}",
            "python3 -m venv .venv",
            ".venv/bin/python -m pip install -r requirements-eda.txt",
            ".venv/bin/python scripts/verify_expression_matrix.py",
            ".venv/bin/python scripts/verify_eda.py",
        ],
    }
    (HANDOFF_DIR / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    checksum_lines = [f"{record['sha256']}  {record['file_name']}" for record in archive_records]
    (EXPORT_DIR / "archive_checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
