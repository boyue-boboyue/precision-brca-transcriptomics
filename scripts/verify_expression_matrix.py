#!/usr/bin/env python3
"""Independently verify expression-matrix files and their manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/expression"
MANIFEST = DATA_DIR / "matrix_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts"].items():
        path = DATA_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Byte-size mismatch: {name}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")

    counts = np.load(DATA_DIR / "counts_uint32.npy", mmap_mode="r")
    tpm = np.load(DATA_DIR / "tpm_float32.npy", mmap_mode="r")
    log_tpm = np.load(DATA_DIR / "log2_tpm_float32.npy", mmap_mode="r")
    expected_shape = (manifest["sample_count"], manifest["gene_count"])
    if counts.shape != expected_shape or tpm.shape != expected_shape:
        raise ValueError("Counts/TPM matrix shape mismatch")
    if log_tpm.shape != expected_shape:
        raise ValueError("Log-TPM matrix shape mismatch")
    if counts.dtype != np.uint32 or tpm.dtype != np.float32 or log_tpm.dtype != np.float32:
        raise ValueError("Unexpected matrix dtype")

    for start in range(0, expected_shape[0], 64):
        stop = min(start + 64, expected_shape[0])
        tpm_chunk = np.asarray(tpm[start:stop])
        log_chunk = np.asarray(log_tpm[start:stop])
        if not np.all(np.isfinite(tpm_chunk)) or np.any(tpm_chunk < 0):
            raise ValueError(f"Invalid TPM values in rows {start}:{stop}")
        if not np.all(np.isfinite(log_chunk)) or np.any(log_chunk < 0):
            raise ValueError(f"Invalid log-TPM values in rows {start}:{stop}")
        expected_log = np.log2(tpm_chunk + np.float32(1.0))
        if not np.array_equal(log_chunk, expected_log):
            raise ValueError(f"Log transform mismatch in rows {start}:{stop}")
        row_sums = tpm_chunk.sum(axis=1, dtype=np.float64)
        if np.any(np.abs(row_sums - 1_000_000) > 1.0):
            raise ValueError(f"Unexpected TPM row sum in rows {start}:{stop}")

    samples = read_tsv(DATA_DIR / "samples.tsv")
    genes = read_tsv(DATA_DIR / "genes.tsv")
    if len(samples) != expected_shape[0] or len(genes) != expected_shape[1]:
        raise ValueError("Axis metadata length mismatch")
    if len({row["case_barcode"] for row in samples}) != len(samples):
        raise ValueError("Case barcodes are not unique")
    if [int(row["matrix_row"]) for row in samples] != list(range(len(samples))):
        raise ValueError("Sample row indices are not contiguous")
    if [int(row["matrix_column"]) for row in genes] != list(range(len(genes))):
        raise ValueError("Gene column indices are not contiguous")

    arrays = {
        "protein": np.load(DATA_DIR / "protein_coding_gene_indices.npy"),
        "labelled": np.load(DATA_DIR / "pam50_labelled_sample_indices.npy"),
        "four_class": np.load(DATA_DIR / "pam50_four_class_sample_indices.npy"),
        "eligible_four_class": np.load(
            DATA_DIR / "pam50_four_class_eligible_sample_indices.npy"
        ),
    }
    if len(arrays["protein"]) != manifest["protein_coding_gene_count"]:
        raise ValueError("Protein-coding index count mismatch")
    if len(arrays["labelled"]) != manifest["labelled_sample_count"]:
        raise ValueError("Labelled sample index count mismatch")
    if len(arrays["four_class"]) != manifest["four_class_sample_count"]:
        raise ValueError("Four-class sample index count mismatch")
    if len(arrays["eligible_four_class"]) != manifest["four_class_eligible_sample_count"]:
        raise ValueError("Eligible four-class sample index count mismatch")
    for name, array in arrays.items():
        if len(np.unique(array)) != len(array):
            raise ValueError(f"Duplicate values in {name} index")
        upper_bound = expected_shape[1] if name == "protein" else expected_shape[0]
        if np.any(array < 0) or np.any(array >= upper_bound):
            raise ValueError(f"Out-of-range values in {name} index")

    class_counts = Counter(
        row["pam50_standard"] for row in samples if row["pam50_standard"]
    )
    if dict(sorted(class_counts.items())) != manifest["class_counts"]:
        raise ValueError("PAM50 class counts do not match manifest")

    result = {
        "status": "PASS",
        "shape": list(expected_shape),
        "counts_dtype": str(counts.dtype),
        "tpm_dtype": str(tpm.dtype),
        "log2_tpm_dtype": str(log_tpm.dtype),
        "protein_coding_gene_count": len(arrays["protein"]),
        "labelled_sample_count": len(arrays["labelled"]),
        "four_class_sample_count": len(arrays["four_class"]),
        "eligible_four_class_sample_count": len(arrays["eligible_four_class"]),
        "class_counts": dict(sorted(class_counts.items())),
    }
    (DATA_DIR / "verification_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

