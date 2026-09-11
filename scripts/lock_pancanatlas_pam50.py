#!/usr/bin/env python3
"""Validate, standardize, match, and lock PanCancer Atlas PAM50 labels."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "brca_tcga_pan_can_atlas_2018"
RAW_DIR = ROOT / "data/raw/pancanatlas/cbioportal" / STUDY_ID
OUTPUT_DIR = ROOT / "data/processed/labels"
GDC_MAP = ROOT / "data/metadata/gdc_star_counts_file_sample_map.tsv"

SUBTYPE_MAP = {
    "BRCA_Basal": "Basal-like",
    "BRCA_Her2": "HER2-enriched",
    "BRCA_LumA": "Luminal A",
    "BRCA_LumB": "Luminal B",
    "BRCA_Normal": "Normal-like",
}
FOUR_CLASS = {"Basal-like", "HER2-enriched", "Luminal A", "Luminal B"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    source_path = RAW_DIR / "pam50_subtype_records.json"
    source_metadata_path = RAW_DIR / "source_metadata.json"
    study_path = RAW_DIR / "study.json"
    attributes_path = RAW_DIR / "clinical_attributes.json"
    inputs = [source_path, source_metadata_path, study_path, attributes_path, GDC_MAP]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    records = load_json(source_path)
    source_metadata = load_json(source_metadata_path)
    study = load_json(study_path)
    attributes = load_json(attributes_path)

    if study.get("studyId") != STUDY_ID:
        raise ValueError("Unexpected cBioPortal study ID")
    subtype_attributes = [
        item for item in attributes if item.get("clinicalAttributeId") == "SUBTYPE"
    ]
    if len(subtype_attributes) != 1 or not subtype_attributes[0].get("patientAttribute"):
        raise ValueError("SUBTYPE must be exactly one patient-level clinical attribute")
    if len(records) != source_metadata.get("record_count"):
        raise ValueError("Source record count does not match source metadata")

    values_by_patient: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.get("studyId") != STUDY_ID:
            raise ValueError(f"Unexpected study ID in record: {record}")
        if record.get("clinicalAttributeId") != "SUBTYPE":
            raise ValueError(f"Unexpected attribute in record: {record}")
        if not record.get("patientAttribute"):
            raise ValueError(f"SUBTYPE record is not patient-level: {record}")
        patient_id = record.get("patientId")
        raw_value = record.get("value")
        if not patient_id or raw_value not in SUBTYPE_MAP:
            raise ValueError(f"Invalid PAM50 record: {record}")
        values_by_patient[patient_id].add(raw_value)

    conflicts = {
        patient_id: sorted(values)
        for patient_id, values in values_by_patient.items()
        if len(values) != 1
    }
    if conflicts:
        raise ValueError(f"Conflicting PAM50 labels: {conflicts}")

    labels = {
        patient_id: next(iter(values))
        for patient_id, values in values_by_patient.items()
    }

    gdc_files: dict[str, set[str]] = defaultdict(set)
    gdc_samples: dict[str, set[str]] = defaultdict(set)
    with GDC_MAP.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            case = row["case_barcode"]
            gdc_files[case].add(row["file_id"])
            gdc_samples[case].add(row["sample_barcode"])

    source_digest = sha256(source_path)
    fields = [
        "case_barcode",
        "pam50_raw",
        "pam50_standard",
        "include_four_class",
        "has_gdc_star_counts",
        "gdc_star_counts_file_count",
        "gdc_sample_barcodes",
        "source_study_id",
        "source_attribute_id",
        "source_scope",
        "source_retrieved_at_utc",
        "source_sha256",
    ]
    rows: list[dict[str, object]] = []
    for patient_id in sorted(labels):
        raw_value = labels[patient_id]
        standard_value = SUBTYPE_MAP[raw_value]
        rows.append(
            {
                "case_barcode": patient_id,
                "pam50_raw": raw_value,
                "pam50_standard": standard_value,
                "include_four_class": int(standard_value in FOUR_CLASS),
                "has_gdc_star_counts": int(patient_id in gdc_files),
                "gdc_star_counts_file_count": len(gdc_files.get(patient_id, set())),
                "gdc_sample_barcodes": ";".join(sorted(gdc_samples.get(patient_id, set()))),
                "source_study_id": STUDY_ID,
                "source_attribute_id": "SUBTYPE",
                "source_scope": "PATIENT",
                "source_retrieved_at_utc": source_metadata["retrieved_at_utc"],
                "source_sha256": source_digest,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    locked_path = OUTPUT_DIR / "pancanatlas_pam50_locked.tsv"
    matched_path = OUTPUT_DIR / "pancanatlas_pam50_gdc_matched.tsv"
    missing_path = OUTPUT_DIR / "gdc_star_counts_cases_without_pam50.tsv"
    lock_path = OUTPUT_DIR / "pancanatlas_pam50_lock.json"

    write_tsv(locked_path, fields, rows)
    matched_rows = [row for row in rows if row["has_gdc_star_counts"] == 1]
    write_tsv(matched_path, fields, matched_rows)
    write_tsv(
        missing_path,
        ["case_barcode", "gdc_star_counts_file_count", "gdc_sample_barcodes"],
        [
            {
                "case_barcode": case,
                "gdc_star_counts_file_count": len(gdc_files[case]),
                "gdc_sample_barcodes": ";".join(sorted(gdc_samples[case])),
            }
            for case in sorted(set(gdc_files) - set(labels))
        ],
    )

    raw_counts = Counter(labels.values())
    standard_counts = Counter(SUBTYPE_MAP[value] for value in labels.values())
    lock = {
        "lock_version": "1.0.0",
        "study_id": STUDY_ID,
        "study_name": study.get("name"),
        "study_import_date": study.get("importDate"),
        "source_provider": "cBioPortal",
        "source_api_root": source_metadata.get("api_root"),
        "source_api_endpoint": source_metadata.get("source_api_endpoint"),
        "source_study_page_url": source_metadata.get("study_page_url"),
        "original_data_url": source_metadata.get("original_data_url"),
        "source_attribute_id": "SUBTYPE",
        "source_scope": "PATIENT",
        "source_retrieved_at_utc": source_metadata["retrieved_at_utc"],
        "source_record_count": len(records),
        "unique_labelled_cases": len(labels),
        "duplicate_source_records": len(records) - len(labels),
        "conflicting_cases": 0,
        "raw_class_counts": dict(sorted(raw_counts.items())),
        "standard_class_counts": dict(sorted(standard_counts.items())),
        "gdc_star_counts_case_count": len(gdc_files),
        "labelled_cases_with_gdc_star_counts": len(matched_rows),
        "labelled_cases_without_gdc_star_counts": len(set(labels) - set(gdc_files)),
        "gdc_star_counts_cases_without_label": len(set(gdc_files) - set(labels)),
        "source_sha256": source_digest,
        "source_metadata_sha256": sha256(source_metadata_path),
        "study_metadata_sha256": sha256(study_path),
        "clinical_attributes_sha256": sha256(attributes_path),
        "gdc_mapping_sha256": sha256(GDC_MAP),
        "locked_table_sha256": sha256(locked_path),
        "matched_table_sha256": sha256(matched_path),
        "missing_labels_table_sha256": sha256(missing_path),
    }
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
