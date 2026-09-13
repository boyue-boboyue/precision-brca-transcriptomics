#!/usr/bin/env python3
"""Verify frozen source downloads and PAM50 cohort provenance without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

METADATA_CHECKSUMS = [
    ROOT / "data/metadata/query_artifacts.sha256",
    ROOT / "data/metadata/metadata_tables.sha256",
]
COHORT_CHECKSUMS = [
    ROOT
    / "data/raw/pancanatlas/cbioportal/brca_tcga_pan_can_atlas_2018/source_files.sha256",
    ROOT / "data/raw/cbioportal/brca_tcga/receptors/source_files.sha256",
]
PUBLICATION_SUPPLEMENT_CHECKSUM = ROOT / "data/metadata/brca_2012_supplement.sha256"
PAM50_LOCK = ROOT / "data/processed/labels/pancanatlas_pam50_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checksum_path(manifest: Path, recorded_path: str) -> Path:
    relative = Path(recorded_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe checksum path in {manifest}: {recorded_path}")
    candidates = (ROOT / relative, manifest.parent / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Checksum target from {manifest.relative_to(ROOT)} is missing: {recorded_path}"
    )


def verify_checksum_manifest(manifest: Path) -> int:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    verified = 0
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"Malformed checksum line {manifest}:{line_number}")
        expected, recorded_path = fields
        path = resolve_checksum_path(manifest, recorded_path.lstrip("*"))
        observed = sha256(path)
        if observed != expected.lower():
            raise RuntimeError(
                f"SHA-256 mismatch for {path.relative_to(ROOT)}: "
                f"expected {expected.lower()}, observed {observed}"
            )
        verified += 1
    if verified == 0:
        raise ValueError(f"No checksum entries found in {manifest}")
    return verified


def verify_pam50_lock() -> dict[str, int]:
    lock = json.loads(PAM50_LOCK.read_text(encoding="utf-8"))
    artifacts = {
        "data/processed/labels/pancanatlas_pam50_locked.tsv": lock[
            "locked_table_sha256"
        ],
        "data/processed/labels/pancanatlas_pam50_gdc_matched.tsv": lock[
            "matched_table_sha256"
        ],
        "data/processed/labels/gdc_star_counts_cases_without_pam50.tsv": lock[
            "missing_labels_table_sha256"
        ],
        "data/metadata/gdc_star_counts_file_sample_map.tsv": lock[
            "gdc_mapping_sha256"
        ],
        "data/raw/pancanatlas/cbioportal/brca_tcga_pan_can_atlas_2018/"
        "pam50_subtype_records.json": lock["source_sha256"],
        "data/raw/pancanatlas/cbioportal/brca_tcga_pan_can_atlas_2018/"
        "study.json": lock["study_metadata_sha256"],
        "data/raw/pancanatlas/cbioportal/brca_tcga_pan_can_atlas_2018/"
        "clinical_attributes.json": lock["clinical_attributes_sha256"],
    }
    for relative, expected in artifacts.items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise RuntimeError(f"PAM50 lock mismatch: {relative}")
    if lock["conflicting_cases"] != 0 or lock["duplicate_source_records"] != 0:
        raise RuntimeError("PAM50 lock contains conflicts or duplicate source records")
    if lock["unique_labelled_cases"] != 981:
        raise RuntimeError("Unexpected locked PAM50 cohort size")
    return {
        "unique_labelled_cases": int(lock["unique_labelled_cases"]),
        "four_class_cases": int(
            lock["unique_labelled_cases"]
            - lock["standard_class_counts"].get("Normal-like", 0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["metadata", "cohort"], required=True)
    parser.add_argument(
        "--skip-unavailable-publication-supplement",
        action="store_true",
        help=(
            "Skip the Git-excluded TCGA 2012 publication supplement while still "
            "verifying all versioned cohort sources and the PAM50 lock"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checksum_manifests = list(METADATA_CHECKSUMS)
    if args.stage == "cohort":
        checksum_manifests.extend(COHORT_CHECKSUMS)
        if not args.skip_unavailable_publication_supplement:
            checksum_manifests.append(PUBLICATION_SUPPLEMENT_CHECKSUM)
    elif args.skip_unavailable_publication_supplement:
        raise ValueError(
            "--skip-unavailable-publication-supplement is valid only for cohort"
        )
    checked_files = sum(verify_checksum_manifest(path) for path in checksum_manifests)
    result: dict[str, object] = {
        "status": "PASS",
        "stage": args.stage,
        "checksum_manifests": [
            str(path.relative_to(ROOT)) for path in checksum_manifests
        ],
        "verified_files": checked_files,
    }
    if args.stage == "cohort":
        result["pam50"] = verify_pam50_lock()
        if args.skip_unavailable_publication_supplement:
            result["skipped_git_excluded_source"] = str(
                PUBLICATION_SUPPLEMENT_CHECKSUM.relative_to(ROOT)
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
