#!/usr/bin/env python3
"""Verify EDA artifacts, hashes, dimensions, and key cohort invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "eda"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    manifest_path = OUT_DIR / "artifact_manifest.json"
    summary_path = OUT_DIR / "eda_summary.json"
    require(manifest_path.exists(), f"Missing {manifest_path}")
    require(summary_path.exists(), f"Missing {summary_path}")
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())

    for artifact in manifest["artifacts"]:
        path = OUT_DIR / artifact["path"]
        require(path.exists(), f"Missing artifact: {path}")
        require(path.stat().st_size == artifact["size_bytes"], f"Size mismatch: {path}")
        require(sha256(path) == artifact["sha256"], f"SHA-256 mismatch: {path}")
    for source in manifest["source_files"]:
        path = ROOT / source["path"]
        require(path.exists(), f"Missing source: {path}")
        require(path.stat().st_size == source["size_bytes"], f"Source size mismatch: {path}")
        require(sha256(path) == source["sha256"], f"Source SHA-256 mismatch: {path}")

    expected_pngs = [
        "01_sample_inclusion_flow.png",
        "02_pam50_class_balance.png",
        "03_sample_expression_and_library_size.png",
        "04_pca_by_pam50.png",
        "05_pca_by_technical_factors.png",
        "06_top500_hvg_clustered_heatmap.png",
        "07_hierarchical_cluster_subtype_correspondence.png",
        "08_receptor_status_by_pam50.png",
        "09_batch_subtype_overlap.png",
        "10_batch_confounding_metrics.png",
    ]
    image_sizes = {}
    for file_name in expected_pngs:
        path = FIG_DIR / file_name
        require(path.exists(), f"Missing figure: {path}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        require(width >= 1200 and height >= 700, f"Unexpectedly small figure: {path}")
        image_sizes[file_name] = [width, height]

    samples = pd.read_csv(TABLE_DIR / "analysis_samples.tsv", sep="\t")
    qc = pd.read_csv(TABLE_DIR / "sample_qc_metrics.tsv", sep="\t")
    balance = pd.read_csv(TABLE_DIR / "class_balance.tsv", sep="\t")
    pca = pd.read_csv(TABLE_DIR / "pca_scores.tsv", sep="\t")
    hvg = pd.read_csv(TABLE_DIR / "top_1000_variable_genes.tsv", sep="\t")
    assignments = pd.read_csv(TABLE_DIR / "hierarchical_cluster_assignments.tsv", sep="\t")
    receptor = pd.read_csv(TABLE_DIR / "receptor_association_stats.tsv", sep="\t")
    batch = pd.read_csv(TABLE_DIR / "batch_subtype_association.tsv", sep="\t")

    require(len(samples) == 1095, "Expected 1,095 analysis-sample records")
    require(len(qc) == 1095, "Expected 1,095 sample QC records")
    require(len(balance) == 9, "Expected five-class plus four-class balance rows")
    require(len(pca) == 981, "Expected 981 PCA samples")
    require(len(hvg) == 1000, "Expected 1,000 high-variable genes")
    require(len(assignments) == 981, "Expected 981 cluster assignments")
    require(assignments["hierarchical_cluster_k5"].nunique() == 5, "Expected five clusters")
    require(set(receptor["marker"]) == {"ER", "PR", "HER2"}, "Receptor markers mismatch")
    require(len(batch) == 5, "Expected five technical-factor audit rows")
    require(
        int(summary["cohorts"]["pam50_five_class"]) == 981
        and int(summary["cohorts"]["pam50_four_class"]) == 945,
        "Cohort counts mismatch",
    )
    require(qc["total_gene_counts"].gt(0).all(), "Non-positive library size")
    require(hvg["variance_rank"].tolist() == list(range(1, 1001)), "HVG ranks are not contiguous")

    report = {
        "status": "PASS",
        "verified_artifact_count": len(manifest["artifacts"]),
        "verified_source_count": len(manifest["source_files"]),
        "figure_count": len(expected_pngs),
        "figure_dimensions": image_sizes,
        "matrix_samples": len(samples),
        "pca_samples": len(pca),
        "hvg_count": len(hvg),
        "hierarchical_clusters": int(assignments["hierarchical_cluster_k5"].nunique()),
        "pam50_five_class": int(summary["cohorts"]["pam50_five_class"]),
        "pam50_four_class": int(summary["cohorts"]["pam50_four_class"]),
    }
    (OUT_DIR / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
