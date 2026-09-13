#!/usr/bin/env python3
"""Reproducible exploratory analysis for TCGA-BRCA PAM50 transcriptomics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_MPL_CONFIG_DIR = Path("/tmp/oncostratify-brca-matplotlib")
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.stats import chi2_contingency
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    completeness_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
EXPR_DIR = ROOT / "data" / "processed" / "expression"
METADATA_DIR = ROOT / "data" / "metadata"
RECEPTOR_DIR = ROOT / "data" / "raw" / "cbioportal" / "brca_tcga" / "receptors"
OUT_DIR = ROOT / "outputs" / "eda"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
SEED = 20260909
N_PERMUTATIONS = 2000

CLASS_ORDER_5 = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched", "Normal-like"]
CLASS_ORDER_4 = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]
SUBTYPE_COLORS = {
    "Luminal A": "#4477AA",
    "Luminal B": "#EE6677",
    "Basal-like": "#228833",
    "HER2-enriched": "#CCBB44",
    "Normal-like": "#AA3377",
    "Unlabelled": "#BBBBBB",
}
STATUS_ORDER = ["Positive", "Negative", "Equivocal", "Indeterminate", "Unknown"]
STATUS_COLORS = {
    "Positive": "#0072B2",
    "Negative": "#D55E00",
    "Equivocal": "#E69F00",
    "Indeterminate": "#999999",
    "Unknown": "#DDDDDD",
}

CJK_FONT_PATHS = [
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
]
CJK_FONT = FontProperties(
    fname=str(next((path for path in CJK_FONT_PATHS if path.exists()), "")) or None
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: str, *, dpi: int = 240, svg: bool = False) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    if svg:
        svg_path = FIG_DIR / f"{stem}.svg"
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
        svg_text = svg_path.read_text()
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
        )
    plt.close(fig)


def set_plot_style() -> None:
    sns.set_theme(context="notebook", style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def read_receptor_map(attribute_id: str) -> dict[str, str]:
    path = RECEPTOR_DIR / f"{attribute_id}.json"
    records = json.loads(path.read_text())
    values: dict[str, str] = {}
    for record in records:
        patient_id = record["patientId"]
        value = record["value"]
        if patient_id in values and values[patient_id] != value:
            raise ValueError(f"Conflicting {attribute_id} values for {patient_id}")
        values[patient_id] = value
    return values


def standardize_receptor(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if text == "positive":
        return "Positive"
    if text == "negative":
        return "Negative"
    if text == "equivocal":
        return "Equivocal"
    if text == "indeterminate":
        return "Indeterminate"
    return "Unknown"


def add_derived_metadata(samples: pd.DataFrame) -> pd.DataFrame:
    result = samples.copy()
    barcode_parts = result["aliquot_barcode"].str.split("-")
    result["tissue_source_site"] = result["case_barcode"].str.split("-").str[1]
    result["rna_plate"] = barcode_parts.str[5]
    result["aliquot_center_code"] = barcode_parts.str[6]

    aliquots = pd.read_csv(METADATA_DIR / "gdc_aliquots.tsv", sep="\t", dtype=str)
    aliquots = aliquots[
        ["aliquot_id", "source_center", "rna_integrity_number"]
    ].drop_duplicates("aliquot_id")
    result = result.merge(aliquots, on="aliquot_id", how="left", validate="one_to_one")

    legacy = pd.read_csv(
        METADATA_DIR / "brca_2012_final_full_sample_summary.tsv", sep="\t", dtype=str
    )
    legacy = legacy.loc[legacy["Tissue"].eq("Tumor"), ["bcr_patient_barcode", "Batch"]]
    legacy = legacy.drop_duplicates("bcr_patient_barcode")
    result = result.merge(
        legacy,
        left_on="case_barcode",
        right_on="bcr_patient_barcode",
        how="left",
        validate="one_to_one",
    ).drop(columns="bcr_patient_barcode")
    result = result.rename(columns={"Batch": "legacy_rnaseq_batch"})

    receptor_sources = {
        "er_status": "ER_STATUS_BY_IHC",
        "pr_status": "PR_STATUS_BY_IHC",
        "her2_status": "IHC_HER2",
    }
    for output_column, attribute_id in receptor_sources.items():
        mapping = read_receptor_map(attribute_id)
        result[output_column] = result["case_barcode"].map(mapping).map(standardize_receptor)

    numeric_columns = [
        "matrix_row",
        "analysis_eligible",
        "include_four_class",
        "total_gene_counts",
        "nonzero_gene_count",
        "tpm_sum",
        "rna_integrity_number",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["analysis_eligible"] = result["analysis_eligible"].eq(1)
    result["include_four_class"] = result["include_four_class"].eq(1)
    return result


def draw_flowchart(samples: pd.DataFrame) -> pd.DataFrame:
    queried_files = int(
        json.loads((METADATA_DIR / "gdc_star_counts_summary.json").read_text())[
            "file_count"
        ]
    )
    selected_cases = len(samples)
    redundant_files = queried_files - selected_cases
    eligible = int(samples["analysis_eligible"].sum())
    labelled = int((samples["analysis_eligible"] & samples["pam50_standard"].notna()).sum())
    four_class = int(
        (
            samples["analysis_eligible"]
            & samples["pam50_standard"].isin(CLASS_ORDER_4)
        ).sum()
    )
    critical_excluded = selected_cases - eligible
    unlabelled_after_eligibility = eligible - labelled
    normal_like_excluded = labelled - four_class

    flow = pd.DataFrame(
        [
            ["GDC STAR Counts files", queried_files, "start"],
            ["Redundant files after one-file-per-case selection", redundant_files, "excluded"],
            ["Unique case-level Primary Tumor samples", selected_cases, "included"],
            ["Critical GDC annotation", critical_excluded, "excluded"],
            ["Analysis-eligible samples", eligible, "included"],
            ["No locked PanCancer Atlas PAM50 label", unlabelled_after_eligibility, "excluded"],
            ["Five-class PAM50 cohort", labelled, "included"],
            ["Normal-like removed from primary endpoint", normal_like_excluded, "excluded"],
            ["Four-class primary cohort", four_class, "included"],
        ],
        columns=["stage", "n", "disposition"],
    )
    flow.to_csv(TABLE_DIR / "sample_inclusion_flow.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    main_nodes = [
        (0.36, 0.91, f"GDC STAR Counts\n{queried_files:,} files"),
        (0.36, 0.73, f"One Primary Tumor file per case\n{selected_cases:,} unique cases"),
        (0.36, 0.55, f"Analysis-eligible samples\n{eligible:,}"),
        (0.36, 0.37, f"Locked five-class PAM50 cohort\n{labelled:,}"),
        (0.36, 0.19, f"Four-class primary analysis cohort\n{four_class:,}"),
    ]
    for x, y, label in main_nodes:
        box = FancyBboxPatch(
            (x - 0.20, y - 0.055),
            0.40,
            0.11,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            facecolor="#DCEAF7",
            edgecolor="#4477AA",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
        )
    exclusions = [
        (0.82, 0.73, f"Excluded: {redundant_files:,}\nRedundant candidate files"),
        (0.82, 0.55, f"Excluded: {critical_excluded:,}\nCritical GDC annotation"),
        (0.82, 0.37, f"Excluded: {unlabelled_after_eligibility:,}\nNo locked PAM50 label"),
        (0.82, 0.19, f"Excluded: {normal_like_excluded:,}\nNormal-like"),
    ]
    for x, y, label in exclusions:
        box = FancyBboxPatch(
            (x - 0.12, y - 0.045),
            0.24,
            0.09,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor="#F6E1DE",
            edgecolor="#BB5566",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
        )
    for idx in range(len(main_nodes) - 1):
        x1, y1, _ = main_nodes[idx]
        x2, y2, _ = main_nodes[idx + 1]
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1 - 0.06),
                (x2, y2 + 0.06),
                arrowstyle="-|>",
                mutation_scale=14,
                color="#555555",
                linewidth=1.3,
            )
        )
    for (_, y, _), (ex, ey, _) in zip(main_nodes[1:], exclusions):
        ax.add_patch(
            FancyArrowPatch(
                (0.565, y),
                (ex - 0.125, ey),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#BB5566",
                linewidth=1.1,
            )
        )
    ax.set_title(
        "TCGA-BRCA sample inclusion and exclusion flow",
        fontsize=15,
        pad=14,
    )
    save_figure(fig, "01_sample_inclusion_flow", svg=True)
    return flow


def plot_class_balance(samples: pd.DataFrame) -> pd.DataFrame:
    labelled = samples.loc[
        samples["analysis_eligible"] & samples["pam50_standard"].notna()
    ].copy()
    records = []
    for cohort, order in [("five_class", CLASS_ORDER_5), ("four_class", CLASS_ORDER_4)]:
        subset = labelled[labelled["pam50_standard"].isin(order)]
        counts = subset["pam50_standard"].value_counts().reindex(order, fill_value=0)
        for subtype, count in counts.items():
            records.append(
                {
                    "cohort": cohort,
                    "subtype": subtype,
                    "count": int(count),
                    "percent": float(100 * count / len(subset)),
                    "cohort_n": int(len(subset)),
                }
            )
    balance = pd.DataFrame(records)
    balance.to_csv(TABLE_DIR / "class_balance.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=False)
    for ax, cohort, order, title in zip(
        axes,
        ["five_class", "four_class"],
        [CLASS_ORDER_5, CLASS_ORDER_4],
        ["五分类 PAM50", "四分类主分析终点"],
    ):
        data = balance[balance["cohort"].eq(cohort)].set_index("subtype").loc[order]
        bars = ax.barh(
            order,
            data["count"],
            color=[SUBTYPE_COLORS[x] for x in order],
            edgecolor="none",
        )
        ax.invert_yaxis()
        ax.set_xlabel("样本数", fontproperties=CJK_FONT)
        ax.set_title(
            f"{title}（n={data['cohort_n'].iloc[0]:,}）",
            fontproperties=CJK_FONT,
        )
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
        for bar, (_, row) in zip(bars, data.iterrows()):
            ax.text(
                bar.get_width() + max(data["count"]) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{int(row['count'])} ({row['percent']:.1f}%)",
                va="center",
                fontsize=9,
            )
        ax.set_xlim(0, max(data["count"]) * 1.30)
    fig.suptitle(
        "PAM50 class balance（类别分布）",
        fontsize=15,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    fig.tight_layout()
    save_figure(fig, "02_pam50_class_balance", svg=True)
    return balance


def calculate_sample_qc(
    samples: pd.DataFrame, log2_tpm: np.ndarray, protein_indices: np.ndarray
) -> pd.DataFrame:
    expression = np.asarray(log2_tpm[:, protein_indices], dtype=np.float32)
    quantile_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    quantiles = np.quantile(expression, quantile_levels, axis=1).T
    qc = samples[
        [
            "matrix_row",
            "case_barcode",
            "sample_barcode",
            "pam50_standard",
            "analysis_eligible",
            "total_gene_counts",
            "nonzero_gene_count",
            "tissue_source_site",
            "rna_plate",
            "legacy_rnaseq_batch",
            "rna_integrity_number",
        ]
    ].copy()
    for index, level in enumerate(quantile_levels):
        qc[f"log2_tpm_q{int(level * 100):02d}"] = quantiles[:, index]
    qc["protein_coding_zero_fraction"] = np.mean(expression == 0, axis=1)
    qc["log10_library_size"] = np.log10(qc["total_gene_counts"].astype(float))
    q1, q3 = qc["log10_library_size"].quantile([0.25, 0.75])
    iqr = q3 - q1
    qc["library_size_tukey_outlier"] = (
        (qc["log10_library_size"] < q1 - 1.5 * iqr)
        | (qc["log10_library_size"] > q3 + 1.5 * iqr)
    )
    qc.to_csv(TABLE_DIR / "sample_qc_metrics.tsv", sep="\t", index=False)

    order = np.argsort(qc["log2_tpm_q50"].to_numpy())
    ordered = qc.iloc[order].reset_index(drop=True)
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 2.0, 0.24], "hspace": 0.12},
    )
    axes[0].fill_between(
        x,
        ordered["log2_tpm_q10"],
        ordered["log2_tpm_q90"],
        color="#9ECAE1",
        alpha=0.45,
        label="10th–90th percentile",
    )
    axes[0].fill_between(
        x,
        ordered["log2_tpm_q25"],
        ordered["log2_tpm_q75"],
        color="#3182BD",
        alpha=0.55,
        label="25th–75th percentile",
    )
    axes[0].plot(x, ordered["log2_tpm_q50"], color="#08306B", linewidth=1.2, label="Median")
    axes[0].set_ylabel("Protein-coding\nlog2(TPM + 1)")
    axes[0].set_title("Per-sample expression distributions")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[0].grid(alpha=0.2)

    outlier = ordered["library_size_tukey_outlier"].to_numpy()
    axes[1].scatter(
        x[~outlier],
        ordered.loc[~outlier, "log10_library_size"],
        s=8,
        alpha=0.65,
        color="#4477AA",
        rasterized=True,
        label="Within Tukey fences",
    )
    axes[1].scatter(
        x[outlier],
        ordered.loc[outlier, "log10_library_size"],
        s=18,
        color="#CC3311",
        marker="x",
        label="Tukey outlier",
    )
    axes[1].axhline(
        ordered["log10_library_size"].median(), color="#222222", linewidth=1, linestyle="--"
    )
    axes[1].set_ylabel("log10 total counts")
    axes[1].set_title("Library size in the same sample order", loc="left", fontsize=11)
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].grid(alpha=0.2)

    subtype_values = ordered["pam50_standard"].fillna("Unlabelled")
    rgba = np.array([matplotlib.colors.to_rgba(SUBTYPE_COLORS[x]) for x in subtype_values])
    axes[2].imshow(rgba[np.newaxis, :, :], aspect="auto", interpolation="nearest")
    axes[2].set_yticks([0], ["PAM50"])
    axes[2].set_xlabel("Samples ordered by median protein-coding expression")
    axes[2].grid(False)
    legend = [
        Patch(facecolor=SUBTYPE_COLORS[x], label=x)
        for x in CLASS_ORDER_5 + ["Unlabelled"]
    ]
    fig.legend(
        handles=legend,
        ncol=6,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.19, hspace=0.16)
    save_figure(fig, "03_sample_expression_and_library_size")
    return qc


def select_hvgs_and_run_pca(
    samples: pd.DataFrame,
    genes: pd.DataFrame,
    log2_tpm: np.ndarray,
    protein_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    cohort_mask = samples["analysis_eligible"] & samples["pam50_standard"].notna()
    cohort = samples.loc[cohort_mask].copy().reset_index(drop=True)
    rows = cohort["matrix_row"].astype(int).to_numpy()
    expression = np.asarray(log2_tpm[rows][:, protein_indices], dtype=np.float32)
    expressed_fraction = np.mean(expression > 1.0, axis=0)
    gene_variance = np.var(expression, axis=0, ddof=1)
    gene_mean = np.mean(expression, axis=0)
    candidate_positions = np.flatnonzero(expressed_fraction >= 0.10)
    if len(candidate_positions) < 1000:
        raise ValueError("Fewer than 1,000 protein-coding genes passed the expression filter")
    ordered_positions = candidate_positions[
        np.argsort(-gene_variance[candidate_positions], kind="stable")
    ]
    top1000_positions = ordered_positions[:1000]
    top1000_matrix_columns = protein_indices[top1000_positions]

    hvg_table = genes.iloc[top1000_matrix_columns].copy()
    hvg_table.insert(0, "variance_rank", np.arange(1, len(hvg_table) + 1))
    hvg_table["mean_log2_tpm"] = gene_mean[top1000_positions]
    hvg_table["variance_log2_tpm"] = gene_variance[top1000_positions]
    hvg_table["fraction_log2_tpm_gt_1"] = expressed_fraction[top1000_positions]
    hvg_table.to_csv(TABLE_DIR / "top_1000_variable_genes.tsv", sep="\t", index=False)

    x_hvg = expression[:, top1000_positions]
    x_mean = x_hvg.mean(axis=0)
    x_sd = x_hvg.std(axis=0, ddof=1)
    if np.any(x_sd == 0):
        raise ValueError("A selected high-variable gene has zero standard deviation")
    x_scaled = (x_hvg - x_mean) / x_sd

    pca = PCA(n_components=20, svd_solver="randomized", random_state=SEED)
    scores = pca.fit_transform(x_scaled)
    score_columns = [f"PC{x}" for x in range(1, 21)]
    score_table = cohort[
        [
            "matrix_row",
            "case_barcode",
            "sample_barcode",
            "pam50_standard",
            "tissue_source_site",
            "rna_plate",
            "aliquot_center_code",
            "source_center",
            "legacy_rnaseq_batch",
            "rna_integrity_number",
            "total_gene_counts",
        ]
    ].copy()
    for index, column in enumerate(score_columns):
        score_table[column] = scores[:, index]
    score_table.to_csv(TABLE_DIR / "pca_scores.tsv", sep="\t", index=False)

    explained = pd.DataFrame(
        {
            "component": score_columns,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    explained.to_csv(TABLE_DIR / "pca_explained_variance.tsv", sep="\t", index=False)

    loadings = hvg_table[
        ["variance_rank", "matrix_column", "gene_id", "gene_name", "gene_type"]
    ].copy()
    for index, column in enumerate(score_columns):
        loadings[column] = pca.components_[index, :]
    loadings.to_csv(TABLE_DIR / "pca_loadings.tsv", sep="\t", index=False)
    return cohort, score_table, explained, x_scaled, top1000_matrix_columns, pca.components_


def add_confidence_ellipse(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, n_std: float = 2.0
) -> None:
    if len(x) < 3:
        return
    covariance = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 0))
    ellipse = Ellipse(
        (np.mean(x), np.mean(y)),
        width,
        height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.10,
        linewidth=1.3,
    )
    ax.add_patch(ellipse)


def plot_pca_by_subtype(score_table: pd.DataFrame, explained: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    for subtype in CLASS_ORDER_5:
        subset = score_table[score_table["pam50_standard"].eq(subtype)]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=24,
            alpha=0.72,
            color=SUBTYPE_COLORS[subtype],
            edgecolors="none",
            rasterized=True,
            label=f"{subtype} (n={len(subset)})",
        )
        add_confidence_ellipse(
            ax,
            subset["PC1"].to_numpy(),
            subset["PC2"].to_numpy(),
            SUBTYPE_COLORS[subtype],
        )
    pc1 = 100 * explained.loc[0, "explained_variance_ratio"]
    pc2 = 100 * explained.loc[1, "explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({pc1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}% variance)")
    ax.set_title("PCA of 1,000 high-variable protein-coding genes")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    save_figure(fig, "04_pca_by_pam50")


def collapse_categories(series: pd.Series, top_n: int) -> pd.Series:
    clean = series.fillna("Unavailable").astype(str)
    counts = clean[clean.ne("Unavailable")].value_counts()
    keep = set(counts.head(top_n).index)
    return clean.map(
        lambda value: value
        if value in keep or value == "Unavailable"
        else f"Other ({max(0, len(counts) - top_n)} levels)"
    )


def categorical_scatter(
    ax: plt.Axes,
    data: pd.DataFrame,
    category: pd.Series,
    title: str,
    top_n: int,
) -> None:
    collapsed = collapse_categories(category, top_n)
    levels = list(collapsed.value_counts().index)
    palette = sns.color_palette("tab20", n_colors=max(1, len(levels)))
    color_map = dict(zip(levels, palette))
    if "Unavailable" in color_map:
        color_map["Unavailable"] = (0.72, 0.72, 0.72)
    for level in levels:
        mask = collapsed.eq(level)
        ax.scatter(
            data.loc[mask, "PC1"],
            data.loc[mask, "PC2"],
            s=15,
            alpha=0.67,
            color=color_map[level],
            edgecolors="none",
            rasterized=True,
            label=f"{level} (n={int(mask.sum())})",
        )
    ax.set_title(title)
    ax.grid(alpha=0.20)
    ax.legend(
        frameon=False,
        fontsize=7.5,
        ncol=2 if len(levels) > 7 else 1,
        loc="best",
    )


def plot_pca_by_technical_factors(
    score_table: pd.DataFrame, explained: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.3), sharex=True, sharey=True)
    categorical_scatter(
        axes[0], score_table, score_table["tissue_source_site"], "Tissue source site (top 10)", 10
    )
    categorical_scatter(axes[1], score_table, score_table["rna_plate"], "RNA plate (top 10)", 10)
    legacy_available = score_table["legacy_rnaseq_batch"].notna()
    categorical_scatter(
        axes[2],
        score_table.loc[legacy_available].reset_index(drop=True),
        score_table.loc[legacy_available, "legacy_rnaseq_batch"].reset_index(drop=True),
        f"Legacy RNA-seq batch (available n={int(legacy_available.sum())})",
        12,
    )
    pc1 = 100 * explained.loc[0, "explained_variance_ratio"]
    pc2 = 100 * explained.loc[1, "explained_variance_ratio"]
    for ax in axes:
        ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pc2:.1f}%)")
    fig.suptitle("PCA coloured by technical or collection factors", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, "05_pca_by_technical_factors")


def adjusted_cramers_v(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    n = table.sum()
    if n <= 1 or table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    rows, columns = table.shape
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    return math.sqrt(phi2_corrected / denominator) if denominator > 0 else float("nan")


def coded_contingency(x_codes: np.ndarray, y_codes: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return np.bincount(x_codes * ny + y_codes, minlength=nx * ny).reshape(nx, ny)


def permutation_cramers_v(
    factor: pd.Series, labels: pd.Series, rng: np.random.Generator
) -> tuple[float, float, float]:
    x_codes, x_levels = pd.factorize(factor.astype(str), sort=True)
    categorical_labels = pd.Categorical(labels, categories=CLASS_ORDER_5)
    y_codes = categorical_labels.codes
    if len(x_levels) < 2 or np.any(y_codes < 0):
        return float("nan"), float("nan"), float("nan")
    observed_table = coded_contingency(x_codes, y_codes, len(x_levels), len(CLASS_ORDER_5))
    observed = adjusted_cramers_v(observed_table)
    chi_square_p = float(chi2_contingency(observed_table, correction=False)[1])
    exceedances = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(y_codes)
        permuted_table = coded_contingency(
            x_codes, shuffled, len(x_levels), len(CLASS_ORDER_5)
        )
        permuted = adjusted_cramers_v(permuted_table)
        if np.isfinite(permuted) and permuted >= observed - 1e-12:
            exceedances += 1
    permutation_p = (exceedances + 1) / (N_PERMUTATIONS + 1)
    return observed, chi_square_p, permutation_p


def categorical_pc_eta_squared(factor: pd.Series, pc_scores: np.ndarray) -> float:
    x = factor.astype(str).to_numpy()
    grand_mean = pc_scores.mean(axis=0)
    total = float(np.sum((pc_scores - grand_mean) ** 2))
    between = 0.0
    for level in np.unique(x):
        group = pc_scores[x == level]
        between += len(group) * float(np.sum((group.mean(axis=0) - grand_mean) ** 2))
    return between / total if total > 0 else float("nan")


def permutation_pc_eta_squared(
    factor: pd.Series, pc_scores: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    observed = categorical_pc_eta_squared(factor, pc_scores)
    values = factor.astype(str).to_numpy()
    exceedances = 0
    for _ in range(N_PERMUTATIONS):
        permuted = categorical_pc_eta_squared(pd.Series(rng.permutation(values)), pc_scores)
        if permuted >= observed - 1e-12:
            exceedances += 1
    return observed, (exceedances + 1) / (N_PERMUTATIONS + 1)


def batch_only_cross_validation(factor: pd.Series, labels: pd.Series) -> tuple[float, float]:
    x = pd.DataFrame({"factor": factor.astype(str).to_numpy()})
    y = labels.astype(str).to_numpy()
    transformer = ColumnTransformer(
        [
            (
                "factor",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                ["factor"],
            )
        ]
    )
    model = Pipeline(
        [
            ("encode", transformer),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    predicted = cross_val_predict(model, x, y, cv=splitter, method="predict", n_jobs=1)
    return float(balanced_accuracy_score(y, predicted)), float(
        f1_score(y, predicted, average="macro")
    )


def analyse_batch_confounding(score_table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    factors = {
        "Tissue source site": "tissue_source_site",
        "RNA plate": "rna_plate",
        "Legacy RNA-seq batch": "legacy_rnaseq_batch",
        "Aliquot center code": "aliquot_center_code",
        "GDC source center": "source_center",
    }
    records = []
    compositions = []
    factor_levels = []
    for display_name, column in factors.items():
        mask = score_table[column].notna()
        factor = score_table.loc[mask, column].astype(str).reset_index(drop=True)
        labels = score_table.loc[mask, "pam50_standard"].reset_index(drop=True)
        pcs = score_table.loc[mask, [f"PC{x}" for x in range(1, 11)]].to_numpy()
        counts = factor.value_counts()
        for level, count in counts.items():
            factor_levels.append(
                {
                    "factor": display_name,
                    "level": level,
                    "count": int(count),
                }
            )
        table = pd.crosstab(factor, labels).reindex(columns=CLASS_ORDER_5, fill_value=0)
        for level, row in table.iterrows():
            for subtype, count in row.items():
                compositions.append(
                    {
                        "factor": display_name,
                        "level": level,
                        "subtype": subtype,
                        "count": int(count),
                        "row_percent": float(100 * count / row.sum()),
                        "level_n": int(row.sum()),
                    }
                )
        if len(counts) > 1:
            cramer_v, chi_square_p, subtype_permutation_p = permutation_cramers_v(
                factor, labels, rng
            )
            ami = float(adjusted_mutual_info_score(labels, factor))
            balanced_accuracy, macro_f1 = batch_only_cross_validation(factor, labels)
            pc_eta_squared, pc_permutation_p = permutation_pc_eta_squared(factor, pcs, rng)
        else:
            cramer_v = chi_square_p = subtype_permutation_p = float("nan")
            ami = balanced_accuracy = macro_f1 = float("nan")
            pc_eta_squared = pc_permutation_p = float("nan")
        records.append(
            {
                "factor": display_name,
                "column": column,
                "n_complete": int(mask.sum()),
                "n_missing": int((~mask).sum()),
                "n_levels": int(len(counts)),
                "minimum_level_n": int(counts.min()) if len(counts) else 0,
                "median_level_n": float(counts.median()) if len(counts) else float("nan"),
                "maximum_level_n": int(counts.max()) if len(counts) else 0,
                "bias_corrected_cramers_v": cramer_v,
                "chi_square_p": chi_square_p,
                "subtype_permutation_p": subtype_permutation_p,
                "adjusted_mutual_information": ami,
                "batch_only_cv_balanced_accuracy": balanced_accuracy,
                "batch_only_cv_macro_f1": macro_f1,
                "pc1_to_pc10_eta_squared": pc_eta_squared,
                "pc_permutation_p": pc_permutation_p,
                "permutations": N_PERMUTATIONS if len(counts) > 1 else 0,
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(TABLE_DIR / "batch_subtype_association.tsv", sep="\t", index=False)
    pd.DataFrame(compositions).to_csv(
        TABLE_DIR / "batch_subtype_composition.tsv", sep="\t", index=False
    )
    pd.DataFrame(factor_levels).to_csv(
        TABLE_DIR / "technical_factor_levels.tsv", sep="\t", index=False
    )
    return result


def plot_batch_overlap(score_table: pd.DataFrame, batch_stats: pd.DataFrame) -> None:
    plot_factors = [
        ("Tissue source site", "tissue_source_site", 14),
        ("RNA plate", "rna_plate", 14),
        ("Legacy RNA-seq batch", "legacy_rnaseq_batch", 12),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 8.4))
    for ax, (display, column, top_n) in zip(axes, plot_factors):
        available = score_table[column].notna()
        data = score_table.loc[available].copy()
        collapsed = collapse_categories(data[column], top_n)
        table = pd.crosstab(collapsed, data["pam50_standard"]).reindex(
            columns=CLASS_ORDER_5, fill_value=0
        )
        table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
        row_percent = table.div(table.sum(axis=1), axis=0) * 100
        ylabels = [f"{level} (n={int(table.loc[level].sum())})" for level in table.index]
        sns.heatmap(
            row_percent,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=100,
            linewidths=0.3,
            linecolor="white",
            cbar=ax is axes[-1],
            cbar_kws={"label": "% within technical-factor level"},
            xticklabels=[x.replace("-", "‑") for x in CLASS_ORDER_5],
            yticklabels=ylabels,
        )
        row = batch_stats[batch_stats["factor"].eq(display)].iloc[0]
        subtitle = (
            f"V={row['bias_corrected_cramers_v']:.3f}; permutation p={row['subtype_permutation_p']:.4f}"
        )
        ax.set_title(f"{display}\n{subtitle}")
        ax.set_xlabel("PAM50 subtype")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Subtype composition across technical or collection factors", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, "09_batch_subtype_overlap")


def plot_batch_metrics(batch_stats: pd.DataFrame) -> None:
    data = batch_stats[batch_stats["n_levels"].gt(1)].copy()
    labels = data["factor"].str.replace("Legacy RNA-seq batch", "Legacy batch", regex=False)
    x = np.arange(len(data))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    metrics = [
        ("bias_corrected_cramers_v", "Subtype association: Cramer's V", "#4477AA"),
        ("batch_only_cv_balanced_accuracy", "Batch-only CV balanced accuracy", "#EE6677"),
        ("pc1_to_pc10_eta_squared", "PC1–10 variance associated (η²)", "#228833"),
    ]
    for offset, (column, display, color) in enumerate(metrics):
        bars = ax.bar(x + (offset - 1) * width, data[column], width, label=display, color=color)
        for bar, value in zip(bars, data[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(0.20, color="#666666", linestyle="--", linewidth=1, label="Five-class chance BA = 0.20")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, min(1, max(0.55, float(data[[x[0] for x in metrics]].max().max()) + 0.12)))
    ax.set_ylabel("Effect or predictive performance (0–1)")
    ax.set_title("Technical-factor overlap with subtype and expression structure")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, "10_batch_confounding_metrics")


def run_hierarchical_analysis(
    cohort: pd.DataFrame,
    x_scaled: np.ndarray,
    genes: pd.DataFrame,
    top1000_matrix_columns: np.ndarray,
) -> dict[str, float]:
    top500 = x_scaled[:, :500]
    sample_linkage = linkage(top500, method="average", metric="correlation")
    gene_linkage = linkage(top500.T, method="average", metric="correlation")
    leaf_order = leaves_list(sample_linkage)
    raw_clusters = fcluster(sample_linkage, t=5, criterion="maxclust")
    ordered_raw_ids = []
    for index in leaf_order:
        cluster_id = int(raw_clusters[index])
        if cluster_id not in ordered_raw_ids:
            ordered_raw_ids.append(cluster_id)
    cluster_map = {raw_id: idx + 1 for idx, raw_id in enumerate(ordered_raw_ids)}
    clusters = np.array([cluster_map[int(x)] for x in raw_clusters])

    assignments = cohort[
        ["matrix_row", "case_barcode", "sample_barcode", "pam50_standard"]
    ].copy()
    assignments["hierarchical_cluster_k5"] = clusters
    assignments["dendrogram_leaf_order"] = np.argsort(leaf_order)
    assignments.to_csv(
        TABLE_DIR / "hierarchical_cluster_assignments.tsv", sep="\t", index=False
    )

    contingency = pd.crosstab(assignments["hierarchical_cluster_k5"], assignments["pam50_standard"])
    contingency = contingency.reindex(index=range(1, 6), columns=CLASS_ORDER_5, fill_value=0)
    contingency.index.name = "hierarchical_cluster_k5"
    contingency.reset_index().to_csv(
        TABLE_DIR / "hierarchical_cluster_contingency.tsv", sep="\t", index=False
    )
    predicted_cluster = assignments["hierarchical_cluster_k5"].to_numpy()
    true_labels = assignments["pam50_standard"].to_numpy()
    purity = float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())
    metrics = {
        "adjusted_rand_index": float(adjusted_rand_score(true_labels, predicted_cluster)),
        "adjusted_mutual_information": float(
            adjusted_mutual_info_score(true_labels, predicted_cluster)
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(true_labels, predicted_cluster)
        ),
        "homogeneity": float(homogeneity_score(true_labels, predicted_cluster)),
        "completeness": float(completeness_score(true_labels, predicted_cluster)),
        "v_measure": float(v_measure_score(true_labels, predicted_cluster)),
        "cluster_purity": purity,
        "bias_corrected_cramers_v": float(adjusted_cramers_v(contingency.to_numpy())),
    }
    pd.DataFrame([metrics]).to_csv(TABLE_DIR / "clustering_metrics.tsv", sep="\t", index=False)

    heat_data = pd.DataFrame(
        np.clip(top500.T, -3, 3),
        index=genes.iloc[top1000_matrix_columns[:500]]["gene_name"].astype(str),
        columns=cohort["sample_barcode"].astype(str),
    )
    column_colors = cohort["pam50_standard"].map(SUBTYPE_COLORS).to_numpy()
    grid = sns.clustermap(
        heat_data,
        row_linkage=gene_linkage,
        col_linkage=sample_linkage,
        col_colors=column_colors,
        cmap="vlag",
        center=0,
        vmin=-3,
        vmax=3,
        xticklabels=False,
        yticklabels=False,
        figsize=(18, 13),
        dendrogram_ratio=(0.12, 0.12),
        colors_ratio=0.025,
        cbar_kws={"label": "Gene-wise z-score"},
    )
    grid.ax_heatmap.set_xlabel(f"Samples (n={len(cohort):,})")
    grid.ax_heatmap.set_ylabel("Top 500 variable protein-coding genes")
    handles = [Patch(facecolor=SUBTYPE_COLORS[x], label=x) for x in CLASS_ORDER_5]
    grid.ax_heatmap.legend(
        handles=handles,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
    )
    grid.fig.suptitle(
        "Hierarchically clustered high-variable gene heatmap",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    grid.fig.savefig(
        FIG_DIR / "06_top500_hvg_clustered_heatmap.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(grid.fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    sns.heatmap(
        contingency,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Samples"},
        ax=axes[0],
    )
    row_percent = contingency.div(contingency.sum(axis=1), axis=0) * 100
    sns.heatmap(
        row_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% within cluster"},
        ax=axes[1],
    )
    axes[0].set_title("Counts")
    axes[1].set_title("Cluster composition (%)")
    for ax in axes:
        ax.set_xlabel("PAM50 subtype")
        ax.set_ylabel("Hierarchical cluster (k=5)")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle(
        f"Hierarchical clusters versus PAM50 (ARI={metrics['adjusted_rand_index']:.3f}, purity={purity:.3f})",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "07_hierarchical_cluster_subtype_correspondence")
    return metrics


def receptor_analysis(samples: pd.DataFrame) -> pd.DataFrame:
    cohort = samples.loc[
        samples["analysis_eligible"] & samples["pam50_standard"].notna()
    ].copy()
    marker_columns = {
        "ER": "er_status",
        "PR": "pr_status",
        "HER2": "her2_status",
    }
    long_records = []
    stats_records = []
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), sharey=True)
    for ax, (marker, column) in zip(axes, marker_columns.items()):
        table = pd.crosstab(cohort["pam50_standard"], cohort[column]).reindex(
            index=CLASS_ORDER_5, columns=STATUS_ORDER, fill_value=0
        )
        for subtype, row in table.iterrows():
            for status, count in row.items():
                long_records.append(
                    {
                        "marker": marker,
                        "subtype": subtype,
                        "status": status,
                        "count": int(count),
                        "row_percent": float(100 * count / row.sum()),
                        "subtype_n": int(row.sum()),
                    }
                )
        row_percent = table.div(table.sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(CLASS_ORDER_5))
        for status in STATUS_ORDER:
            values = row_percent[status].to_numpy()
            ax.bar(
                CLASS_ORDER_5,
                values,
                bottom=bottom,
                color=STATUS_COLORS[status],
                label=status,
                width=0.75,
            )
            bottom += values
        binary = cohort[cohort[column].isin(["Positive", "Negative"])]
        binary_table = pd.crosstab(binary["pam50_standard"], binary[column]).reindex(
            index=CLASS_ORDER_5, columns=["Positive", "Negative"], fill_value=0
        )
        chi2, p_value, _, _ = chi2_contingency(binary_table, correction=False)
        stats_records.append(
            {
                "marker": marker,
                "cohort_n": int(len(cohort)),
                "known_n": int(cohort[column].ne("Unknown").sum()),
                "positive_or_negative_n": int(len(binary)),
                "coverage_percent": float(100 * cohort[column].ne("Unknown").mean()),
                "chi_square_positive_negative": float(chi2),
                "chi_square_p": float(p_value),
                "bias_corrected_cramers_v": float(
                    adjusted_cramers_v(binary_table.to_numpy())
                ),
            }
        )
        ax.set_ylim(0, 100)
        ax.set_title(
            f"{marker} (known {int(cohort[column].ne('Unknown').sum())}/{len(cohort)})"
        )
        ax.set_xlabel("PAM50 subtype")
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", alpha=0.22)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Samples within subtype (%)")
    handles = [Patch(facecolor=STATUS_COLORS[x], label=x) for x in STATUS_ORDER]
    axes[-1].legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Clinical receptor status by PAM50 subtype", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, "08_receptor_status_by_pam50")
    pd.DataFrame(long_records).to_csv(
        TABLE_DIR / "receptor_pam50_contingency.tsv", sep="\t", index=False
    )
    stats = pd.DataFrame(stats_records)
    stats.to_csv(TABLE_DIR / "receptor_association_stats.tsv", sep="\t", index=False)
    return stats


def build_summary(
    samples: pd.DataFrame,
    flow: pd.DataFrame,
    balance: pd.DataFrame,
    qc: pd.DataFrame,
    explained: pd.DataFrame,
    clustering_metrics: dict[str, float],
    receptor_stats: pd.DataFrame,
    batch_stats: pd.DataFrame,
) -> dict:
    labelled = samples["analysis_eligible"] & samples["pam50_standard"].notna()
    four_class = samples["analysis_eligible"] & samples["pam50_standard"].isin(CLASS_ORDER_4)
    nonconstant_batch = batch_stats[batch_stats["n_levels"].gt(1)].copy()
    strongest_overlap = nonconstant_batch.loc[
        nonconstant_batch["bias_corrected_cramers_v"].idxmax()
    ]
    strongest_batch_predictor = nonconstant_batch.loc[
        nonconstant_batch["batch_only_cv_balanced_accuracy"].idxmax()
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "random_seed": SEED,
        "permutations": N_PERMUTATIONS,
        "cohorts": {
            "matrix_samples": int(len(samples)),
            "analysis_eligible": int(samples["analysis_eligible"].sum()),
            "pam50_five_class": int(labelled.sum()),
            "pam50_four_class": int(four_class.sum()),
        },
        "class_counts_five": {
            key: int(value)
            for key, value in samples.loc[labelled, "pam50_standard"]
            .value_counts()
            .reindex(CLASS_ORDER_5)
            .items()
        },
        "expression_qc": {
            "library_size_min": int(qc["total_gene_counts"].min()),
            "library_size_median": float(qc["total_gene_counts"].median()),
            "library_size_max": int(qc["total_gene_counts"].max()),
            "library_size_tukey_outliers": int(qc["library_size_tukey_outlier"].sum()),
            "median_protein_coding_zero_fraction": float(
                qc["protein_coding_zero_fraction"].median()
            ),
        },
        "pca": {
            "cohort_n": int(labelled.sum()),
            "hvg_count": 1000,
            "pc1_explained_percent": float(
                100 * explained.loc[0, "explained_variance_ratio"]
            ),
            "pc2_explained_percent": float(
                100 * explained.loc[1, "explained_variance_ratio"]
            ),
            "pc1_to_pc10_cumulative_percent": float(
                100 * explained.loc[9, "cumulative_explained_variance_ratio"]
            ),
        },
        "clustering": clustering_metrics,
        "receptor_association": receptor_stats.set_index("marker").to_dict(orient="index"),
        "technical_batch": {
            "strongest_subtype_overlap_factor": strongest_overlap["factor"],
            "strongest_bias_corrected_cramers_v": float(
                strongest_overlap["bias_corrected_cramers_v"]
            ),
            "strongest_overlap_permutation_p": float(
                strongest_overlap["subtype_permutation_p"]
            ),
            "strongest_batch_only_predictor": strongest_batch_predictor["factor"],
            "strongest_batch_only_cv_balanced_accuracy": float(
                strongest_batch_predictor["batch_only_cv_balanced_accuracy"]
            ),
            "five_class_chance_balanced_accuracy": 0.2,
            "invariant_factors": batch_stats.loc[
                batch_stats["n_levels"].eq(1), "factor"
            ].tolist(),
            "factor_results": batch_stats.to_dict(orient="records"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
    }


def write_report(summary: dict) -> None:
    batch = summary["technical_batch"]
    pca = summary["pca"]
    qc = summary["expression_qc"]
    clustering = summary["clustering"]
    receptor = summary["receptor_association"]
    strongest_ba = batch["strongest_batch_only_cv_balanced_accuracy"]
    if strongest_ba >= 0.35:
        batch_interpretation = (
            "技术标签能够在交叉验证中明显预测 PAM50，存在有意义的重合；后续模型必须做按技术组分层/分组的敏感性分析。"
        )
    elif strongest_ba >= 0.25:
        batch_interpretation = (
            "存在一定技术标签与 PAM50 重合，但批次单独预测能力远低于完整表达信号；尚不构成近乎完全的标签泄漏。"
        )
    else:
        batch_interpretation = (
            "批次单独预测 PAM50 的能力接近五分类机会水平，没有看到明显的技术标签泄漏。"
        )
    report = f"""# TCGA-BRCA transcriptomic exploratory analysis

生成时间（UTC）：{summary['generated_at_utc']}  
随机种子：{summary['random_seed']}  
主要分析队列：{summary['cohorts']['pam50_five_class']:,} 个五分类样本；{summary['cohorts']['pam50_four_class']:,} 个四分类样本。

## 主要结论

- 表达矩阵最终包含 {summary['cohorts']['matrix_samples']:,} 个病例级 Primary Tumor 样本；经 GDC 关键注释与锁定 PAM50 标签筛选后，五分类队列为 {summary['cohorts']['pam50_five_class']:,}，四分类主队列为 {summary['cohorts']['pam50_four_class']:,}。
- 文库规模中位数为 {qc['library_size_median']:,.0f} counts，范围 {qc['library_size_min']:,}–{qc['library_size_max']:,}；按 log10 文库规模的 Tukey 规则标记 {qc['library_size_tukey_outliers']} 个探索性异常值。异常标记不是自动排除标准。
- 基于 1,000 个高变蛋白编码基因的 PCA：PC1 解释 {pca['pc1_explained_percent']:.2f}%，PC2 解释 {pca['pc2_explained_percent']:.2f}%，前10个主成分累计解释 {pca['pc1_to_pc10_cumulative_percent']:.2f}% 变异。
- 用前500个高变基因进行相关距离/平均连接层次聚类，并切为5群：ARI={clustering['adjusted_rand_index']:.3f}，AMI={clustering['adjusted_mutual_information']:.3f}，purity={clustering['cluster_purity']:.3f}。这衡量无监督表达结构与 PAM50 的对应程度，不应当作分类性能。
- ER、PR、HER2 与 PAM50 的阳性/阴性关联 Cramer's V 分别为 {receptor['ER']['bias_corrected_cramers_v']:.3f}、{receptor['PR']['bias_corrected_cramers_v']:.3f}、{receptor['HER2']['bias_corrected_cramers_v']:.3f}；覆盖率分别为 {receptor['ER']['coverage_percent']:.1f}%、{receptor['PR']['coverage_percent']:.1f}%、{receptor['HER2']['coverage_percent']:.1f}%。
- 技术/采集标签与 PAM50 的最大校正 Cramer's V 为 {batch['strongest_bias_corrected_cramers_v']:.3f}（{batch['strongest_subtype_overlap_factor']}，置换 p={batch['strongest_overlap_permutation_p']:.4f}）。最强的批次单变量五折交叉验证 balanced accuracy 为 {strongest_ba:.3f}（{batch['strongest_batch_only_predictor']}；五分类机会水平0.20）。{batch_interpretation}
- Aliquot center code 与 GDC source center 在该 RNA 队列中均为常数，无法解释样本间表达或 subtype 差异。

## 图表

1. [样本纳入排除流程](figures/01_sample_inclusion_flow.png)
2. [四类及五类 class balance](figures/02_pam50_class_balance.png)
3. [逐样本表达分布与文库规模](figures/03_sample_expression_and_library_size.png)
4. [PCA：按 PAM50 subtype 着色](figures/04_pca_by_pam50.png)
5. [PCA：按 TSS、RNA plate 与 legacy batch 着色](figures/05_pca_by_technical_factors.png)
6. [前500个高变基因层次聚类热图](figures/06_top500_hvg_clustered_heatmap.png)
7. [层次聚类与 PAM50 对应关系](figures/07_hierarchical_cluster_subtype_correspondence.png)
8. [ER、PR、HER2 与 PAM50 列联图](figures/08_receptor_status_by_pam50.png)
9. [技术标签内的 PAM50 组成](figures/09_batch_subtype_overlap.png)
10. [批次混杂效应量与批次单变量预测](figures/10_batch_confounding_metrics.png)

## 方法口径

- PCA 与聚类仅使用五分类、analysis-eligible 样本；先保留至少10%样本满足 log2(TPM+1)>1 的蛋白编码基因，再按全队列方差选取前1,000个并进行基因内 z-score 标准化。
- 聚类热图使用前500个高变基因、相关距离和平均连接；k=5 仅用于和五个 PAM50 类别作描述性对应。
- 批次重合检查同时报告校正 Cramer's V、2,000次标签置换检验、批次单变量五折交叉验证，以及技术标签与 PC1–PC10 的总体 eta-squared。PCA 图为了可读性仅单独着色高频技术水平，但统计检验使用全部原始水平。
- ER/PR/HER2 来自 cBioPortal `brca_tcga` 病例级临床记录，通过 TCGA patient barcode 与锁定的 PanCancer Atlas PAM50 标签连接；HER2 使用 `IHC_HER2`。阳性/阴性关联检验排除 Unknown、Equivocal 和 Indeterminate，但列联图保留这些类别。

## 批次结论的边界

TSS 同时反映样本来源、患者构成和收集路径，并不是纯实验批次。显著关联只说明分布不独立，不能单独证明技术伪影。后续建模建议至少报告：按 plate/TSS 分组的敏感性验证、训练折内基因筛选，以及模型预测与 plate/TSS 的残余关联。

## 数据表

所有逐样本、PCA、HVG、聚类、受体列联和批次检验结果均位于 [`tables/`](tables/)。`eda_summary.json` 保存机器可读摘要，`artifact_manifest.json` 保存输出文件哈希。
"""
    (OUT_DIR / "eda_report.md").write_text(report)


def write_manifest(summary: dict) -> None:
    artifacts = []
    for path in sorted(OUT_DIR.rglob("*")):
        if not path.is_file() or path.name in {"artifact_manifest.json", "verification_report.json"}:
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(OUT_DIR)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    source_files = [
        EXPR_DIR / "matrix_manifest.json",
        ROOT / "data" / "processed" / "labels" / "pancanatlas_pam50_lock.json",
        RECEPTOR_DIR / "ER_STATUS_BY_IHC.json",
        RECEPTOR_DIR / "PR_STATUS_BY_IHC.json",
        RECEPTOR_DIR / "IHC_HER2.json",
        RECEPTOR_DIR / "source_metadata.json",
        RECEPTOR_DIR / "source_files.sha256",
    ]
    manifest = {
        "generated_at_utc": summary["generated_at_utc"],
        "script": "scripts/run_eda.py",
        "random_seed": SEED,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "source_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in source_files
        ],
    }
    (OUT_DIR / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def json_safe(value):
    """Convert NumPy scalars and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    required_receptors = [
        RECEPTOR_DIR / "ER_STATUS_BY_IHC.json",
        RECEPTOR_DIR / "PR_STATUS_BY_IHC.json",
        RECEPTOR_DIR / "IHC_HER2.json",
        RECEPTOR_DIR / "source_files.sha256",
    ]
    missing = [str(path) for path in required_receptors if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing locked receptor inputs. Run scripts/fetch_cbioportal_receptors.sh first: "
            + ", ".join(missing)
        )

    samples = pd.read_csv(EXPR_DIR / "samples.tsv", sep="\t", dtype=str)
    genes = pd.read_csv(EXPR_DIR / "genes.tsv", sep="\t", dtype={"matrix_column": int})
    samples = add_derived_metadata(samples)
    samples.to_csv(TABLE_DIR / "analysis_samples.tsv", sep="\t", index=False)
    log2_tpm = np.load(EXPR_DIR / "log2_tpm_float32.npy", mmap_mode="r")
    protein_indices = np.load(EXPR_DIR / "protein_coding_gene_indices.npy")

    flow = draw_flowchart(samples)
    balance = plot_class_balance(samples)
    qc = calculate_sample_qc(samples, log2_tpm, protein_indices)
    cohort, score_table, explained, x_scaled, top1000_columns, _ = select_hvgs_and_run_pca(
        samples, genes, log2_tpm, protein_indices
    )
    plot_pca_by_subtype(score_table, explained)
    plot_pca_by_technical_factors(score_table, explained)
    clustering_metrics = run_hierarchical_analysis(
        cohort, x_scaled, genes, top1000_columns
    )
    receptor_stats = receptor_analysis(samples)
    batch_stats = analyse_batch_confounding(score_table)
    plot_batch_overlap(score_table, batch_stats)
    plot_batch_metrics(batch_stats)

    summary = build_summary(
        samples,
        flow,
        balance,
        qc,
        explained,
        clustering_metrics,
        receptor_stats,
        batch_stats,
    )
    summary = json_safe(summary)
    (OUT_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    write_report(summary)
    write_manifest(summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
