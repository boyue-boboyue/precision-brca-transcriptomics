# TCGA-BRCA transcriptomic exploratory analysis

生成时间（UTC）：2026-09-09T14:50:15Z  
随机种子：20260909  
主要分析队列：981 个五分类样本；945 个四分类样本。

## 主要结论

- 表达矩阵最终包含 1,095 个病例级 Primary Tumor 样本；经 GDC 关键注释与锁定 PAM50 标签筛选后，五分类队列为 981，四分类主队列为 945。
- 文库规模中位数为 57,971,543 counts，范围 19,303,211–114,015,705；按 log10 文库规模的 Tukey 规则标记 18 个探索性异常值。异常标记不是自动排除标准。
- 基于 1,000 个高变蛋白编码基因的 PCA：PC1 解释 18.14%，PC2 解释 8.23%，前10个主成分累计解释 46.29% 变异。
- 用前500个高变基因进行相关距离/平均连接层次聚类，并切为5群：ARI=0.296，AMI=0.408，purity=0.680。这衡量无监督表达结构与 PAM50 的对应程度，不应当作分类性能。
- ER、PR、HER2 与 PAM50 的阳性/阴性关联 Cramer's V 分别为 0.813、0.720、0.440；覆盖率分别为 95.4%、95.3%、83.0%。
- 技术/采集标签与 PAM50 的最大校正 Cramer's V 为 0.117（RNA plate，置换 p=0.0125）。最强的批次单变量五折交叉验证 balanced accuracy 为 0.288（RNA plate；五分类机会水平0.20）。存在一定技术标签与 PAM50 重合，但批次单独预测能力远低于完整表达信号；尚不构成近乎完全的标签泄漏。
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
