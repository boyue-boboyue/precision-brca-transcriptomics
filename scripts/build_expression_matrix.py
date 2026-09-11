#!/usr/bin/env python3
"""Build case-level TCGA-BRCA counts and TPM matrices from GDC STAR Counts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw/gdc/star_counts"
FILE_MAP = ROOT / "data/metadata/gdc_star_counts_file_sample_map.tsv"
ANNOTATIONS = ROOT / "data/metadata/gdc_annotations.json"
LABELS = ROOT / "data/processed/labels/pancanatlas_pam50_locked.tsv"
GDC_STATUS = ROOT / "data/metadata/gdc_status.json"
OUTPUT_DIR = ROOT / "data/processed/expression"

CRITICAL_CATEGORIES = {
    "Biospecimen identity unknown",
    "Genotype mismatch",
    "Item does not meet study protocol",
    "Item flagged DNU",
    "Item is noncanonical",
    "Subject identity unknown",
}
ADVISORY_CATEGORIES = {
    "History of unacceptable prior treatment related to a prior/other malignancy",
    "Neoadjuvant therapy",
    "Prior malignancy",
}

EXPRESSION_COLUMNS = [
    "gene_id",
    "gene_name",
    "gene_type",
    "unstranded",
    "tpm_unstranded",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def annotation_index() -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["data"]["hits"]:
        if annotation.get("status") != "Approved":
            continue
        keys = [
            annotation.get("entity_id"),
            annotation.get("entity_submitter_id"),
        ]
        # Only a case-level annotation applies to the entire case. For sample,
        # aliquot, or file annotations, indexing by case would incorrectly flag
        # unrelated biospecimens belonging to the same patient.
        if annotation.get("entity_type") == "case":
            keys.extend(
                [annotation.get("case_id"), annotation.get("case_submitter_id")]
            )
        for key in keys:
            if key:
                index[str(key)].append(annotation)
    return index


def annotations_for_candidate(
    candidate: dict[str, Any], index: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = {
        candidate["file_id"],
        candidate["case_id"],
        candidate["case_barcode"],
        candidate["sample_id"],
        candidate["sample_barcode"],
        candidate["aliquot_id"],
        candidate["aliquot_barcode"],
    }
    found: dict[str, dict[str, Any]] = {}
    for key in keys:
        if not key:
            continue
        for annotation in index.get(key, []):
            found[annotation["annotation_id"]] = annotation

    critical = [
        item for item in found.values() if item.get("category") in CRITICAL_CATEGORIES
    ]
    advisory = [
        item for item in found.values() if item.get("category") in ADVISORY_CATEGORIES
    ]
    return (
        sorted(critical, key=lambda item: item["annotation_id"]),
        sorted(advisory, key=lambda item: item["annotation_id"]),
    )


def load_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    annotation_lookup = annotation_index()
    for row in read_tsv(FILE_MAP):
        candidate: dict[str, Any] = dict(row)
        candidate["file_size"] = int(row["file_size"])
        candidate["path"] = RAW_ROOT / row["file_id"] / row["file_name"]
        if not candidate["path"].is_file():
            raise FileNotFoundError(candidate["path"])
        critical, advisory = annotations_for_candidate(candidate, annotation_lookup)
        candidate["critical_annotations"] = critical
        candidate["advisory_annotations"] = advisory
        candidates.append(candidate)
    if len({item["file_id"] for item in candidates}) != len(candidates):
        raise ValueError("Duplicate file IDs in GDC file-to-sample mapping")
    return candidates


def read_expression(
    path: Path,
    reference_gene_ids: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, float | int]]:
    frame = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        usecols=EXPRESSION_COLUMNS,
        dtype={"gene_id": "string", "gene_name": "string", "gene_type": "string"},
    )
    frame = frame.loc[frame["gene_id"].str.startswith("ENSG", na=False)].reset_index(drop=True)
    gene_ids = frame["gene_id"].to_numpy(dtype=str)
    if reference_gene_ids is not None and not np.array_equal(gene_ids, reference_gene_ids):
        raise ValueError(f"Gene order or identifiers differ in {path}")

    counts64 = pd.to_numeric(frame["unstranded"], errors="raise").to_numpy(dtype=np.int64)
    tpm64 = pd.to_numeric(frame["tpm_unstranded"], errors="raise").to_numpy(dtype=np.float64)
    if np.any(counts64 < 0) or np.any(counts64 > np.iinfo(np.uint32).max):
        raise ValueError(f"Counts outside uint32 range in {path}")
    if not np.all(np.isfinite(tpm64)) or np.any(tpm64 < 0):
        raise ValueError(f"Invalid TPM values in {path}")

    counts = counts64.astype(np.uint32, copy=False)
    tpm = tpm64.astype(np.float32, copy=False)
    metrics: dict[str, float | int] = {
        "total_gene_counts": int(counts64.sum()),
        "nonzero_gene_count": int(np.count_nonzero(counts64)),
        "tpm_sum": float(tpm64.sum()),
        "max_gene_count": int(counts64.max()),
        "max_tpm": float(tpm64.max()),
    }
    return frame[["gene_id", "gene_name", "gene_type"]], counts, tpm, metrics


def join_annotation_field(items: list[dict[str, Any]], field: str) -> str:
    return ";".join(sorted({str(item.get(field, "")) for item in items if item.get(field)}))


def choose_case_files(
    candidates: list[dict[str, Any]], metrics: dict[str, dict[str, float | int]]
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_case[candidate["case_barcode"]].append(candidate)

    duplicate_candidates = [
        item for group in by_case.values() if len(group) > 1 for item in group
    ]
    print(
        f"Computing selection QC for {len(duplicate_candidates)} files "
        f"from {sum(len(group) > 1 for group in by_case.values())} duplicate cases...",
        flush=True,
    )
    for number, candidate in enumerate(sorted(duplicate_candidates, key=lambda x: x["file_id"]), 1):
        _, _, _, file_metrics = read_expression(candidate["path"])
        metrics[candidate["file_id"]] = file_metrics
        if number % 10 == 0 or number == len(duplicate_candidates):
            print(f"  duplicate QC {number}/{len(duplicate_candidates)}", flush=True)

    selected: list[dict[str, Any]] = []
    for case_barcode in sorted(by_case):
        group = by_case[case_barcode]
        if len(group) == 1:
            winner = group[0]
            winner["selection_reason"] = "only_primary_tumor_file"
        else:
            winner = min(
                group,
                key=lambda item: (
                    len(item["critical_annotations"]) > 0,
                    -int(metrics[item["file_id"]]["total_gene_counts"]),
                    -int(metrics[item["file_id"]]["nonzero_gene_count"]),
                    item["file_id"],
                ),
            )
            winner["selection_reason"] = (
                "prefer_no_critical_annotation_then_highest_total_counts_"
                "then_nonzero_genes_then_file_id"
            )
        selected.append(winner)
        for item in group:
            item["selected"] = int(item is winner)
    return selected


def load_labels() -> dict[str, dict[str, str]]:
    labels = {row["case_barcode"]: row for row in read_tsv(LABELS)}
    if len(labels) != 981:
        raise ValueError(f"Expected 981 locked PAM50 labels, found {len(labels)}")
    return labels


def build_matrices() -> None:
    candidates = load_candidates()
    if len(candidates) != 1111:
        raise ValueError(f"Expected 1111 GDC files, found {len(candidates)}")

    metrics: dict[str, dict[str, float | int]] = {}
    selected = choose_case_files(candidates, metrics)
    if len(selected) != 1095:
        raise ValueError(f"Expected 1095 unique cases, found {len(selected)}")

    labels = load_labels()
    first_meta, first_counts, first_tpm, first_metrics = read_expression(selected[0]["path"])
    reference_gene_ids = first_meta["gene_id"].to_numpy(dtype=str)
    gene_count = len(first_meta)
    sample_count = len(selected)
    if gene_count != 60660:
        raise ValueError(f"Expected 60660 GENCODE genes, found {gene_count}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="matrix-build-", dir=OUTPUT_DIR) as temp_name:
        temp_dir = Path(temp_name)
        counts_path = temp_dir / "counts_uint32.npy"
        tpm_path = temp_dir / "tpm_float32.npy"
        log_path = temp_dir / "log2_tpm_float32.npy"

        counts_matrix = np.lib.format.open_memmap(
            counts_path, mode="w+", dtype=np.uint32, shape=(sample_count, gene_count)
        )
        tpm_matrix = np.lib.format.open_memmap(
            tpm_path, mode="w+", dtype=np.float32, shape=(sample_count, gene_count)
        )

        sample_rows: list[dict[str, Any]] = []
        tpm_sums: list[float] = []
        max_observed_count = 0
        for index, candidate in enumerate(selected):
            if index == 0:
                gene_meta, counts, tpm, file_metrics = (
                    first_meta,
                    first_counts,
                    first_tpm,
                    first_metrics,
                )
            else:
                gene_meta, counts, tpm, file_metrics = read_expression(
                    candidate["path"], reference_gene_ids
                )
            if not np.array_equal(gene_meta["gene_id"].to_numpy(dtype=str), reference_gene_ids):
                raise ValueError(f"Gene metadata mismatch in {candidate['path']}")

            counts_matrix[index] = counts
            tpm_matrix[index] = tpm
            metrics[candidate["file_id"]] = file_metrics
            tpm_sums.append(float(file_metrics["tpm_sum"]))
            max_observed_count = max(max_observed_count, int(file_metrics["max_gene_count"]))

            label = labels.get(candidate["case_barcode"])
            sample_rows.append(
                {
                    "matrix_row": index,
                    "case_id": candidate["case_id"],
                    "case_barcode": candidate["case_barcode"],
                    "sample_id": candidate["sample_id"],
                    "sample_barcode": candidate["sample_barcode"],
                    "aliquot_id": candidate["aliquot_id"],
                    "aliquot_barcode": candidate["aliquot_barcode"],
                    "file_id": candidate["file_id"],
                    "file_name": candidate["file_name"],
                    "selection_reason": candidate["selection_reason"],
                    "analysis_eligible": int(not candidate["critical_annotations"]),
                    "critical_annotation_ids": join_annotation_field(
                        candidate["critical_annotations"], "annotation_id"
                    ),
                    "critical_annotation_categories": join_annotation_field(
                        candidate["critical_annotations"], "category"
                    ),
                    "advisory_annotation_ids": join_annotation_field(
                        candidate["advisory_annotations"], "annotation_id"
                    ),
                    "advisory_annotation_categories": join_annotation_field(
                        candidate["advisory_annotations"], "category"
                    ),
                    "total_gene_counts": file_metrics["total_gene_counts"],
                    "nonzero_gene_count": file_metrics["nonzero_gene_count"],
                    "tpm_sum": f"{float(file_metrics['tpm_sum']):.6f}",
                    "pam50_raw": label["pam50_raw"] if label else "",
                    "pam50_standard": label["pam50_standard"] if label else "",
                    "include_four_class": label["include_four_class"] if label else "0",
                }
            )
            if (index + 1) % 50 == 0 or index + 1 == sample_count:
                print(f"  matrix rows {index + 1}/{sample_count}", flush=True)

        counts_matrix.flush()
        tpm_matrix.flush()
        del counts_matrix
        del tpm_matrix

        tpm_read = np.load(tpm_path, mmap_mode="r")
        log_matrix = np.lib.format.open_memmap(
            log_path, mode="w+", dtype=np.float32, shape=(sample_count, gene_count)
        )
        for start in range(0, sample_count, 64):
            stop = min(start + 64, sample_count)
            np.log2(tpm_read[start:stop] + np.float32(1.0), out=log_matrix[start:stop])
        log_matrix.flush()
        del log_matrix
        del tpm_read

        genes_path = temp_dir / "genes.tsv"
        gene_rows = []
        for index, row in first_meta.iterrows():
            gene_id = str(row["gene_id"])
            gene_type = "" if pd.isna(row["gene_type"]) else str(row["gene_type"])
            gene_rows.append(
                {
                    "matrix_column": index,
                    "gene_id": gene_id,
                    "gene_id_without_version": gene_id.split(".", 1)[0],
                    "gene_name": "" if pd.isna(row["gene_name"]) else str(row["gene_name"]),
                    "gene_type": gene_type,
                    "is_protein_coding": int(gene_type == "protein_coding"),
                }
            )
        write_tsv(
            genes_path,
            [
                "matrix_column",
                "gene_id",
                "gene_id_without_version",
                "gene_name",
                "gene_type",
                "is_protein_coding",
            ],
            gene_rows,
        )

        samples_path = temp_dir / "samples.tsv"
        sample_fields = list(sample_rows[0])
        write_tsv(samples_path, sample_fields, sample_rows)

        protein_indices = np.flatnonzero(
            first_meta["gene_type"].fillna("").to_numpy(dtype=str) == "protein_coding"
        ).astype(np.int32)
        np.save(temp_dir / "protein_coding_gene_indices.npy", protein_indices)

        labelled_indices = np.array(
            [row["matrix_row"] for row in sample_rows if row["pam50_standard"]],
            dtype=np.int32,
        )
        four_class_indices = np.array(
            [row["matrix_row"] for row in sample_rows if row["include_four_class"] == "1"],
            dtype=np.int32,
        )
        eligible_four_class_indices = np.array(
            [
                row["matrix_row"]
                for row in sample_rows
                if row["include_four_class"] == "1" and row["analysis_eligible"] == 1
            ],
            dtype=np.int32,
        )
        np.save(temp_dir / "pam50_labelled_sample_indices.npy", labelled_indices)
        np.save(temp_dir / "pam50_four_class_sample_indices.npy", four_class_indices)
        np.save(
            temp_dir / "pam50_four_class_eligible_sample_indices.npy",
            eligible_four_class_indices,
        )

        candidate_fields = [
            "case_barcode",
            "sample_barcode",
            "aliquot_barcode",
            "file_id",
            "file_name",
            "selected",
            "selection_reason",
            "critical_annotation_ids",
            "critical_annotation_categories",
            "advisory_annotation_ids",
            "advisory_annotation_categories",
            "total_gene_counts",
            "nonzero_gene_count",
            "tpm_sum",
        ]
        candidate_rows = []
        for candidate in sorted(candidates, key=lambda item: (item["case_barcode"], item["file_id"])):
            file_metrics = metrics[candidate["file_id"]]
            candidate_rows.append(
                {
                    "case_barcode": candidate["case_barcode"],
                    "sample_barcode": candidate["sample_barcode"],
                    "aliquot_barcode": candidate["aliquot_barcode"],
                    "file_id": candidate["file_id"],
                    "file_name": candidate["file_name"],
                    "selected": candidate["selected"],
                    "selection_reason": candidate.get("selection_reason", "not_selected"),
                    "critical_annotation_ids": join_annotation_field(
                        candidate["critical_annotations"], "annotation_id"
                    ),
                    "critical_annotation_categories": join_annotation_field(
                        candidate["critical_annotations"], "category"
                    ),
                    "advisory_annotation_ids": join_annotation_field(
                        candidate["advisory_annotations"], "annotation_id"
                    ),
                    "advisory_annotation_categories": join_annotation_field(
                        candidate["advisory_annotations"], "category"
                    ),
                    "total_gene_counts": file_metrics["total_gene_counts"],
                    "nonzero_gene_count": file_metrics["nonzero_gene_count"],
                    "tpm_sum": f"{float(file_metrics['tpm_sum']):.6f}",
                }
            )
        write_tsv(temp_dir / "candidate_file_audit.tsv", candidate_fields, candidate_rows)

        gdc_status = json.loads(GDC_STATUS.read_text(encoding="utf-8"))
        class_counts = Counter(
            row["pam50_standard"] for row in sample_rows if row["pam50_standard"]
        )
        eligible_class_counts = Counter(
            row["pam50_standard"]
            for row in sample_rows
            if row["pam50_standard"] and row["analysis_eligible"] == 1
        )
        manifest = {
            "manifest_version": "1.0.0",
            "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "builder_script_sha256": sha256(Path(__file__)),
            "gdc_file_sample_map_sha256": sha256(FILE_MAP),
            "gdc_annotations_sha256": sha256(ANNOTATIONS),
            "pam50_locked_labels_sha256": sha256(LABELS),
            "axis_order": "rows=samples; columns=genes",
            "case_selection": "one Primary Tumor STAR Counts file per TCGA case",
            "sample_count": sample_count,
            "gene_count": gene_count,
            "protein_coding_gene_count": int(len(protein_indices)),
            "labelled_sample_count": int(len(labelled_indices)),
            "four_class_sample_count": int(len(four_class_indices)),
            "four_class_eligible_sample_count": int(len(eligible_four_class_indices)),
            "analysis_eligible_sample_count": sum(
                row["analysis_eligible"] == 1 for row in sample_rows
            ),
            "critical_annotation_sample_count": sum(
                row["analysis_eligible"] == 0 for row in sample_rows
            ),
            "class_counts": dict(sorted(class_counts.items())),
            "eligible_class_counts": dict(sorted(eligible_class_counts.items())),
            "gdc_data_release": gdc_status.get("data_release"),
            "gdc_api_commit": gdc_status.get("commit"),
            "gene_model": "GENCODE v36",
            "counts_source_column": "unstranded",
            "tpm_source_column": "tpm_unstranded",
            "log_transform": "log2(tpm_unstranded + 1)",
            "counts_dtype": "uint32",
            "tpm_dtype": "float32",
            "log2_tpm_dtype": "float32",
            "max_observed_gene_count": max_observed_count,
            "minimum_tpm_row_sum": min(tpm_sums),
            "maximum_tpm_row_sum": max(tpm_sums),
            "critical_annotation_categories": sorted(CRITICAL_CATEGORIES),
            "advisory_annotation_categories": sorted(ADVISORY_CATEGORIES),
            "selection_priority": [
                "no critical GDC annotation",
                "highest total gene counts",
                "highest nonzero gene count",
                "lexicographically smallest file UUID",
            ],
            "pre_model_filtering": "none; expression/variance filters must be fit inside CV",
        }

        artifact_names = [
            "counts_uint32.npy",
            "tpm_float32.npy",
            "log2_tpm_float32.npy",
            "genes.tsv",
            "samples.tsv",
            "protein_coding_gene_indices.npy",
            "pam50_labelled_sample_indices.npy",
            "pam50_four_class_sample_indices.npy",
            "pam50_four_class_eligible_sample_indices.npy",
            "candidate_file_audit.tsv",
        ]
        manifest["artifacts"] = {
            name: {
                "bytes": (temp_dir / name).stat().st_size,
                "sha256": sha256(temp_dir / name),
            }
            for name in artifact_names
        }
        (temp_dir / "matrix_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for name in artifact_names + ["matrix_manifest.json"]:
            os.replace(temp_dir / name, OUTPUT_DIR / name)

    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    build_matrices()
