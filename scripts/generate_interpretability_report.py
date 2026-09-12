#!/usr/bin/env python3
"""Generate interpretation figures and an integrated biological report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "interpretability"
BIO_DIR = OUTPUT_DIR / "biological_validation"
FIGURE_DIR = OUTPUT_DIR / "figures"
DISTRIBUTION_DIR = FIGURE_DIR / "stable_gene_distributions"
REPORT_PATH = OUTPUT_DIR / "interpretability_report.md"
CLASS_ORDER = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]
CLASS_SHORT = {
    "Luminal A": "LumA",
    "Luminal B": "LumB",
    "Basal-like": "Basal",
    "HER2-enriched": "HER2E",
}
CLASS_COLORS = {
    "Luminal A": "#3B82F6",
    "Luminal B": "#8B5CF6",
    "Basal-like": "#EF4444",
    "HER2-enriched": "#F59E0B",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_style() -> None:
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
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_elastic_class_coefficients(class_top: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for axis, subtype in zip(axes.flat, CLASS_ORDER):
        group = class_top.loc[class_top["pam50_class"].eq(subtype)]
        selected = pd.concat(
            [
                group.loc[group["direction"].eq("negative")].head(6),
                group.loc[group["direction"].eq("positive")].head(6),
            ]
        ).sort_values("mean_coefficient_including_zeros")
        colors = np.where(
            selected["mean_coefficient_including_zeros"].gt(0), "#2563EB", "#D97706"
        )
        axis.barh(
            selected["gene_name"],
            selected["mean_coefficient_including_zeros"],
            color=colors,
        )
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set_title(f"{subtype}: positive and negative coefficients")
        axis.set_xlabel("Mean standardized Elastic-net coefficient (missing = 0)")
        for container in axis.containers:
            axis.bar_label(container, fmt="%.3f", padding=2, fontsize=7)
    fig.suptitle("Class-level Elastic-net signals across five outer folds", fontsize=15)
    name = "01_elastic_net_class_coefficients.png"
    save_figure(fig, name)
    return name


def plot_rf_permutation(permutation: pd.DataFrame) -> str:
    selected = permutation.head(25).sort_values(
        "macro_f1_importance_mean_including_unselected_zeros"
    )
    fig, axis = plt.subplots(figsize=(10, 9))
    values = selected["macro_f1_importance_mean_including_unselected_zeros"]
    axis.barh(selected["gene_name"], values, color=np.where(values.ge(0), "#0F766E", "#B45309"))
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("Mean held-out macro-F1 decrease after permutation")
    axis.set_ylabel("")
    axis.set_title("Random-forest held-out permutation importance\n(mean across five outer validation folds)")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.4f", padding=2, fontsize=7)
    name = "02_rf_heldout_permutation_importance.png"
    save_figure(fig, name)
    return name


def plot_stability(elastic: pd.DataFrame, forest: pd.DataFrame) -> str:
    elastic_pivot = elastic.pivot_table(
        index="gene_name",
        columns="pam50_class",
        values="top20_outer_fold_frequency",
        aggfunc="max",
        fill_value=0,
    ).reindex(columns=CLASS_ORDER, fill_value=0)
    elastic_pivot.columns = [f"Elastic {CLASS_SHORT[column]}" for column in elastic_pivot]
    forest_series = forest.set_index("gene_name")["top20_outer_fold_frequency"].rename(
        "RF permutation"
    )
    matrix = elastic_pivot.join(forest_series, how="outer").fillna(0)
    matrix = matrix.assign(maximum=matrix.max(axis=1), total=matrix.sum(axis=1)).sort_values(
        ["maximum", "total"], ascending=False
    ).drop(columns=["maximum", "total"])
    matrix = matrix.head(35)
    fig_height = max(7, 0.28 * len(matrix) + 2)
    fig, axis = plt.subplots(figsize=(9, fig_height))
    sns.heatmap(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        cbar_kws={"label": "Fraction of outer folds in top 20"},
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title("Top-20 feature stability across outer folds")
    name = "03_top20_cross_fold_stability.png"
    save_figure(fig, name)
    return name


def plot_expression_overview(distribution: pd.DataFrame, stable: pd.DataFrame) -> str:
    medians = distribution.pivot(
        index="gene_name", columns="pam50_class", values="median_log2_tpm"
    ).reindex(columns=CLASS_ORDER)
    maximum = stable.set_index("gene_name")["maximum_top20_fold_count"]
    medians = medians.loc[maximum.sort_values(ascending=False).index.intersection(medians.index)]
    z = medians.sub(medians.mean(axis=1), axis=0).div(
        medians.std(axis=1, ddof=0).replace(0, 1), axis=0
    )
    fig_height = max(7, 0.27 * len(z) + 2)
    fig, axis = plt.subplots(figsize=(8.5, fig_height))
    sns.heatmap(
        z,
        cmap="vlag",
        center=0,
        annot=len(z) <= 28,
        fmt=".1f",
        linewidths=0.35,
        cbar_kws={"label": "Within-gene z-score of subtype median"},
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title("Stable-gene expression patterns in development cases")
    name = "04_stable_gene_subtype_expression_overview.png"
    save_figure(fig, name)
    return name


def plot_individual_distributions(long: pd.DataFrame, stable: pd.DataFrame) -> list[str]:
    names: list[str] = []
    order = stable.sort_values(["maximum_top20_fold_count", "gene_name"], ascending=[False, True])[
        "gene_name"
    ]
    for gene in order:
        group = long.loc[long["gene_name"].eq(gene)]
        fig, axis = plt.subplots(figsize=(7.5, 4.8))
        sns.violinplot(
            data=group,
            x="pam50_class",
            y="log2_tpm",
            order=CLASS_ORDER,
            palette=CLASS_COLORS,
            inner="box",
            cut=0,
            linewidth=0.8,
            hue="pam50_class",
            legend=False,
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel("log2(TPM + 1)")
        axis.set_title(f"{gene}: expression distribution by PAM50 subtype")
        axis.tick_params(axis="x", rotation=18)
        filename = f"{str(gene).lower().replace('.', '_')}.png"
        fig.savefig(
            DISTRIBUTION_DIR / filename,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        names.append(f"stable_gene_distributions/{filename}")
    return names


def plot_receptor_associations(receptor: pd.DataFrame) -> str:
    pivot = receptor.pivot(
        index="gene_name", columns="receptor", values="rank_biserial_correlation"
    ).reindex(columns=["ER", "PR", "HER2"])
    q = receptor.pivot(index="gene_name", columns="receptor", values="bh_q_value").reindex(
        columns=["ER", "PR", "HER2"]
    )
    order = pivot.abs().max(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    q = q.loc[order]
    annotations = pivot.applymap(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    for gene in pivot.index:
        for marker in pivot.columns:
            if pd.notna(q.loc[gene, marker]) and q.loc[gene, marker] < 0.05:
                annotations.loc[gene, marker] += "*"
    fig_height = max(7, 0.27 * len(pivot) + 2)
    fig, axis = plt.subplots(figsize=(7.5, fig_height))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annotations,
        fmt="",
        linewidths=0.4,
        cbar_kws={"label": "Rank-biserial correlation (Positive vs Negative)"},
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title("Stable-gene association with ER/PR/HER2 IHC status\n* BH q < 0.05")
    name = "05_stable_gene_receptor_associations.png"
    save_figure(fig, name)
    return name


def plot_enrichment(enrichment: pd.DataFrame) -> str:
    primary_all = enrichment.loc[
        enrichment["analysis"].eq("primary_union_background")
        & enrichment["adjusted_p_value"].notna()
    ].copy()
    primary = (
        primary_all.sort_values("adjusted_p_value")
        .groupby("source", as_index=False, group_keys=False)
        .head(10)
        .copy()
    )
    if primary.empty:
        fig, axis = plt.subplots(figsize=(9, 3))
        axis.text(0.5, 0.5, "No GO:BP or Reactome terms passed BH-FDR 0.05", ha="center", va="center")
        axis.axis("off")
    else:
        primary["minus_log10_q"] = -np.log10(primary["adjusted_p_value"].clip(lower=1e-300))
        primary["label"] = primary["term_name"].str.slice(0, 65)
        primary = primary.sort_values("minus_log10_q")
        fig, axis = plt.subplots(figsize=(11, max(6, 0.38 * len(primary) + 1.5)))
        colors = primary["source"].map({"GO:BP": "#2563EB", "REAC": "#D97706"}).fillna("#64748B")
        axis.barh(primary["label"], primary["minus_log10_q"], color=colors)
        axis.set_xlabel("−log10(BH-adjusted p-value)")
        axis.set_ylabel("")
        axis.set_title("Stable-gene functional enrichment with expression-filter background")
        handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2563EB", markersize=9, label="GO:BP"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#D97706", markersize=9, label="Reactome"),
        ]
        axis.legend(handles=handles, loc="lower right")
    name = "06_functional_enrichment.png"
    save_figure(fig, name)
    return name


def plot_shap_cases(top: pd.DataFrame, summary: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)
    summary_index = summary.set_index("display_order")
    for axis, (display_order, group) in zip(axes.flat, top.groupby("display_order", sort=True)):
        case = summary_index.loc[display_order]
        selected = group.nsmallest(10, "absolute_rank").sort_values("shap_value")
        colors = np.where(selected["shap_value"].ge(0), "#2563EB", "#D97706")
        labels = [
            f"{gene}  ({value:.2f})"
            for gene, value in zip(selected["gene_name"], selected["raw_log2_tpm"])
        ]
        axis.barh(labels, selected["shap_value"], color=colors)
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set_xlabel(f"SHAP contribution to P({case['predicted']})")
        status = "correct" if case["correct"] else "error"
        axis.set_title(
            f"{case['case_barcode']} · {case['selection_role']}\n"
            f"observed {case['observed']} → predicted {case['predicted']} ({status}), p={case['confidence']:.3f}"
        )
    fig.suptitle("Individual TreeSHAP explanations (top absolute contributions)", fontsize=15)
    name = "07_individual_shap_explanations.png"
    save_figure(fig, name)
    return name


def markdown_table(frame: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    work = frame[columns].head(n) if n is not None else frame[columns]
    headers = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in work.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.4g}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([headers, separator, *rows])


def write_report(
    elastic_top: pd.DataFrame,
    rf: pd.DataFrame,
    stable: pd.DataFrame,
    classification: pd.DataFrame,
    subtype: pd.DataFrame,
    overlap: pd.DataFrame,
    receptor: pd.DataFrame,
    enrichment: pd.DataFrame,
    shap_summary: pd.DataFrame,
    shap_top: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    class_lines = []
    for subtype_name in CLASS_ORDER:
        group = elastic_top.loc[elastic_top["pam50_class"].eq(subtype_name)]
        positive = ", ".join(group.loc[group["direction"].eq("positive")].head(5)["gene_name"])
        negative = ", ".join(group.loc[group["direction"].eq("negative")].head(5)["gene_name"])
        class_lines.append(f"- **{subtype_name}**：positive `{positive}`；negative `{negative}`。")
    rf_top = rf.head(10)[
        ["gene_name", "macro_f1_importance_mean_including_unselected_zeros", "top20_outer_fold_count"]
    ]
    category_counts = classification["evidence_category"].value_counts().rename_axis(
        "evidence_category"
    ).reset_index(name="gene_count")
    significant_receptor = receptor.loc[
        receptor["bh_q_value"].lt(0.05)
        & receptor["rank_biserial_correlation"].abs().ge(0.33)
        & receptor["gene_name"].isin(
            classification.loc[
                classification["evidence_category"].eq(
                    "clinically_correlated_non_PAM50"
                ),
                "gene_name",
            ]
        )
    ].copy()
    primary_enrichment = enrichment.loc[
        enrichment["analysis"].eq("primary_union_background")
    ].sort_values("adjusted_p_value").groupby(
        "source", as_index=False, group_keys=False
    ).head(5)
    stable_n = len(stable)
    pam50_n = int(overlap["is_locked_pam50_gene"].sum())
    expected_n = int(overlap["is_predeclared_expected_marker"].sum())
    max_residual = float(shap_summary["absolute_additivity_residual"].max())
    report = f"""# OncoStratify-BRCA 解释性与生物学验证报告

生成时间：`{now_utc()}`

## 结论摘要

- 全局与类别级解释基于五个已锁定外层折；Elastic-net系数是标准化输入下的系数，未选择的基因按0计入跨折平均。
- 随机森林permutation importance只在对应outer-validation病例上计算，每折重复{config['permutation_importance']['repeats']}次；没有加载locked-test表达用于该分析。
- 按“任一Elastic-net类别或随机森林permutation进入top 20至少3/5折”的预锁定规则，共得到`{stable_n}`个稳定重要基因。其中`{pam50_n}`个与锁定PAM50 50基因重叠，`{expected_n}`个属于预先列出的可检查标志基因。
- 个体解释使用在全部756例development上重拟合的冻结random forest；SHAP background为100例预锁定development病例。只解释6例按预测类别、正确性和置信度预选的locked-test病例，不重新计算性能或选择模型。最大概率重构残差为`{max_residual:.3g}`。

## 1. 全局解释：held-out permutation importance

下表给出按五个outer-validation折平均macro-F1下降排序的前10个基因。负值不解释为保护作用，只表示有限验证样本与相关特征下的置换噪声。

{markdown_table(rf_top, list(rf_top.columns))}

![Held-out permutation importance](figures/02_rf_heldout_permutation_importance.png)

## 2. 类别级解释：Elastic-net positive/negative genes

{chr(10).join(class_lines)}

![Elastic-net class coefficients](figures/01_elastic_net_class_coefficients.png)

完整结果见`elastic_net_class_top_genes.tsv`与`elastic_net_coefficient_summary.tsv.gz`。

## 3. 跨折稳定性

主要稳定性指标为每个基因在五个外层折中进入top 20的次数/频率；同时保存top 50和top 100指示列。高相关基因会分摊系数或置换重要性，因此排名第一不等于唯一关键基因，也不代表因果作用。

![Cross-fold stability](figures/03_top20_cross_fold_stability.png)

## 4. 稳定基因的表达与PAM50重叠

所有表达分布与检验只使用development病例。每个稳定基因均保存逐病例log2(TPM+1)、四亚型的均值/中位数/IQR，以及Kruskal-Wallis检验和BH q值；每个基因另有一张violin图。

![Stable-gene subtype expression](figures/04_stable_gene_subtype_expression_overview.png)

PAM50重叠使用已锁定的50基因表和其中记录的历史符号映射；它是标签内生性审计，不用于提高基因排名。

## 5. ER/PR/HER2临床状态交叉验证

仅纳入有明确Positive/Negative IHC状态的development病例。效应量为“阳性组相对阴性组”的rank-biserial correlation；星号代表同一受体内BH q<0.05。

![Receptor cross-validation](figures/05_stable_gene_receptor_associations.png)

满足`q<0.05`且`|rank-biserial|≥0.33`的非PAM50临床相关结果：

{markdown_table(significant_receptor, ['receptor', 'gene_name', 'positive_n', 'negative_n', 'rank_biserial_correlation', 'bh_q_value'], n=20) if not significant_receptor.empty else '没有结果同时达到预设显著性与效应量阈值。'}

## 6. GO Biological Process与Reactome富集

目标集为上述稳定基因。主背景是五个outer-training低表达过滤器中至少一折通过的全部蛋白编码基因；另用5/5折均通过的基因作为背景敏感性分析。g:Profiler以`domain_scope=custom`执行GO:BP与Reactome过表达检验，并使用BH-FDR阈值0.05。

{markdown_table(primary_enrichment, ['source', 'term_id', 'term_name', 'adjusted_p_value', 'intersection_size'], n=10) if not primary_enrichment.empty else '主背景分析没有通路通过BH-FDR 0.05。'}

![Functional enrichment](figures/06_functional_enrichment.png)

## 7. 已知标志、相关基因与候选基因

分类是可复现的操作性标签，不是穷尽式文献裁定：

- `known_marker_or_PAM50`：预先指定的已知检查基因或锁定PAM50成员；
- `clinically_correlated_non_PAM50`：不属于前一类，但与至少一个IHC受体达到`q<0.05`且效应量绝对值≥0.33；
- `potential_candidate_not_novelty_claim`：其余稳定且亚型表达差异BH q<0.05的基因，需要独立队列和功能实验验证；
- `other_model_associated`：其余稳定模型关联基因。

{markdown_table(category_counts, ['evidence_category', 'gene_count'])}

完整逐基因判定见`biological_validation/stable_gene_evidence_classification.tsv`。“潜在候选”明确不等于新颖性、机制或临床效用结论。

## 8. 个体级TreeSHAP

蓝色条提高预测类别概率，橙色条降低预测类别概率；基因名后的括号是该病例原始log2(TPM+1)。SHAP值是在训练集background条件下对模型输出的归因，不是基因干预效应。

![Individual SHAP explanations](figures/07_individual_shap_explanations.png)

病例选择与概率审计：

{markdown_table(shap_summary, ['selection_role', 'case_barcode', 'observed', 'predicted', 'correct', 'confidence', 'probability_margin'])}

完整四类别、全部2,000个模型输入的SHAP值见`shap_values_selected_cases.tsv.gz`。

## 9. 方法学边界

- PAM50标签由表达信号生成，因此本项目主要验证模型复现既有分型的能力，不是独立发现亚型。
- 高维相关性会在Elastic-net系数、permutation importance与SHAP贡献之间分摊信号；单基因排名不可视为唯一机制。
- TCGA是单一回顾性队列；临床受体交叉验证是内部一致性检查，不能替代外部验证或前瞻性临床评价。
- 预期信号（ESR1、PGR、ERBB2、FOXA1、GATA3、MKI67、KRT5、KRT14、KRT17、EGFR）只被审计，没有强制进入特征集、稳定列表或图表。

## 10. 主要方法来源

- Parker et al., 2009, PAM50 classifier: https://doi.org/10.1200/JCO.2008.18.1370
- TCGA Network, 2012, molecular portraits of human breast tumours: https://doi.org/10.1038/nature11412
- Lundberg et al., 2020, TreeSHAP: https://doi.org/10.1038/s42256-019-0138-9
- Kolberg et al., 2023, g:Profiler and custom-background enrichment: https://doi.org/10.1093/nar/gkad347
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild presentation artifacts from unchanged result tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if FIGURE_DIR.exists() and any(FIGURE_DIR.iterdir()) and not args.rebuild:
        raise RuntimeError("Interpretability figure directory is not empty")
    if REPORT_PATH.exists() and not args.rebuild:
        raise RuntimeError("Interpretability report already exists")
    main_manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    bio_manifest = json.loads((BIO_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    if main_manifest.get("status") != "COMPLETE" or bio_manifest.get("status") != "COMPLETE":
        raise RuntimeError("Interpretability and biological validation must be complete")
    config = json.loads((ROOT / "config/interpretability_v1.json").read_text(encoding="utf-8"))
    elastic_top = pd.read_csv(OUTPUT_DIR / "elastic_net_class_top_genes.tsv", sep="\t")
    elastic_stability = pd.read_csv(OUTPUT_DIR / "elastic_net_top20_stability.tsv", sep="\t")
    rf = pd.read_csv(OUTPUT_DIR / "rf_permutation_importance_summary.tsv.gz", sep="\t")
    rf_stability = pd.read_csv(OUTPUT_DIR / "rf_top20_stability.tsv", sep="\t")
    stable = pd.read_csv(BIO_DIR / "stable_important_genes.tsv", sep="\t")
    long = pd.read_csv(BIO_DIR / "stable_gene_expression_by_development_sample.tsv.gz", sep="\t")
    distribution = pd.read_csv(BIO_DIR / "stable_gene_subtype_expression_summary.tsv", sep="\t")
    subtype = pd.read_csv(BIO_DIR / "stable_gene_subtype_tests.tsv", sep="\t")
    overlap = pd.read_csv(BIO_DIR / "stable_gene_pam50_overlap.tsv", sep="\t")
    receptor = pd.read_csv(BIO_DIR / "stable_gene_receptor_associations.tsv", sep="\t")
    classification = pd.read_csv(BIO_DIR / "stable_gene_evidence_classification.tsv", sep="\t")
    enrichment = pd.read_csv(BIO_DIR / "functional_enrichment_results.tsv", sep="\t")
    shap_summary = pd.read_csv(OUTPUT_DIR / "shap_case_summary.tsv", sep="\t")
    shap_top = pd.read_csv(OUTPUT_DIR / "shap_predicted_class_top_contributions.tsv", sep="\t")

    set_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=args.rebuild)
    DISTRIBUTION_DIR.mkdir(parents=True, exist_ok=args.rebuild)
    figure_names = [
        plot_elastic_class_coefficients(elastic_top),
        plot_rf_permutation(rf),
        plot_stability(elastic_stability, rf_stability),
        plot_expression_overview(distribution, stable),
        plot_receptor_associations(receptor),
        plot_enrichment(enrichment),
        plot_shap_cases(shap_top, shap_summary),
    ]
    distribution_names = plot_individual_distributions(long, stable)
    write_report(
        elastic_top,
        rf,
        stable,
        classification,
        subtype,
        overlap,
        receptor,
        enrichment,
        shap_summary,
        shap_top,
        config,
    )
    artifacts = [REPORT_PATH, *[FIGURE_DIR / name for name in figure_names], *[FIGURE_DIR / name for name in distribution_names]]
    report_manifest = {
        "schema_version": "1.0.0",
        "status": "COMPLETE",
        "created_at_utc": now_utc(),
        "stable_gene_distribution_figure_n": len(distribution_names),
        "main_figure_n": len(figure_names),
        "reporter_sha256": sha256(Path(__file__).resolve()),
        "input_sha256": {
            "outputs/interpretability/run_manifest.json": sha256(OUTPUT_DIR / "run_manifest.json"),
            "outputs/interpretability/biological_validation/run_manifest.json": sha256(BIO_DIR / "run_manifest.json"),
        },
        "output_sha256": {
            str(path.relative_to(OUTPUT_DIR)): sha256(path) for path in artifacts
        },
    }
    (OUTPUT_DIR / "report_manifest.json").write_text(
        json.dumps(report_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "main_figures": len(figure_names),
                "stable_gene_distribution_figures": len(distribution_names),
                "report": str(REPORT_PATH.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
