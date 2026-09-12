# 稳健性与局限性分析

## 分析边界

本分析是模型选择完成后的 development-only 确认实验。随机森林结构和2,000基因训练折方差筛选规则固定，不重新调参；所有低表达过滤、插补、方差筛选与标准化均只在当前训练折拟合。locked test 表达行加载数为0，也没有生成新的测试集预测或性能指标。

## 稳健性结果

| display_name | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | balanced_accuracy_std | delta_macro_f1_mean_vs_reference |
| --- | --- | --- | --- | --- | --- |
| 4-class · TPM · primary | 0.905 | 0.038 | 0.905 | 0.041 | 0.000 |
| 5-class · TPM · primary | 0.817 | 0.051 | 0.800 | 0.048 | -0.088 |
| 4-class · PAM50 excluded | 0.899 | 0.036 | 0.900 | 0.036 | -0.005 |
| 4-class · log2 CPM | 0.911 | 0.031 | 0.911 | 0.032 | 0.007 |
| 4-class · TPM≥0.5 in 10% | 0.909 | 0.039 | 0.908 | 0.042 | 0.004 |
| 4-class · TPM≥1 in 20% | 0.906 | 0.043 | 0.906 | 0.048 | 0.001 |
| 4-class · TPM≥5 in 10% | 0.906 | 0.042 | 0.904 | 0.048 | 0.001 |
| 4-class · unweighted | 0.878 | 0.038 | 0.853 | 0.040 | -0.026 |
| 4-class · PAM50 only | 0.917 | 0.034 | 0.920 | 0.038 | 0.012 |
| 4-class · grouped seed 20260909 | 0.905 | 0.034 | 0.905 | 0.037 | -0.000 |
| 4-class · grouped seed 20260910 | 0.906 | 0.017 | 0.907 | 0.023 | 0.001 |
| 4-class · grouped seed 20260911 | 0.907 | 0.014 | 0.906 | 0.026 | 0.002 |
| 4-class · sample split seed 20260909 | 0.905 | 0.038 | 0.905 | 0.041 | 0.000 |

参考方案（四分类、log2(TPM+1)、PAM50 included、balanced、锁定病例折）的 macro F1 为 `0.905 ± 0.038`，balanced accuracy 为 `0.905 ± 0.041`。表中差值均相对该参考方案；五分类与不同随机折因病例集合或折分配不同，差值只能作描述性比较。

随机种子敏感性（种子同时改变病例级fold分配和随机森林random_state）中，mean macro F1 范围为 `0.905–0.907`，mean balanced accuracy 范围为 `0.905–0.907`。

病例拆分确认实验使用同一 development 队列和种子：病例分组方案 macro F1 `0.905`，样本分层方案 `0.905`。当前矩阵每病例最多一个样本，因此两种方案均没有病例跨折重叠；这个实验验证了实现完整性，但不能估计存在重复样本时样本级拆分导致的乐观偏倚。

## 标签来源一致性

| source_a | source_b | overlap_cases | agreement_cases | discordant_cases | exact_concordance | cohen_kappa | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_cBioPortal_PanCancer_SUBTYPE | locked_PanCancer_table | 981 | 981 | 0 | 1.0 | 1.0 | Serialization and standardization integrity check |
| locked_PanCancer_table | expression_sample_axis | 981 | 981 | 0 | 1.0 | 1.0 | Label propagation to the expression matrix |
| locked_PanCancer_table | five_class_split_table | 981 | 981 | 0 | 1.0 | 1.0 | Label propagation to all five-class assignments |
| locked_PanCancer_table | four_class_split_table | 945 | 945 | 0 | 1.0 | 1.0 | Label propagation after excluding Normal-like |
| TCGA_2012_publication_PAM50 | PanCancer_Atlas_PAM50 | 447 | 398 | 49 | 0.8903803131991052 | 0.8379248340622016 | Independent historical data freeze; discordance is scientifically meaningful, not a serialization failure |

原始cBioPortal记录、锁定标签表、表达矩阵样本轴和split表之间为完全一致。独立的TCGA 2012 publication freeze与PanCancer Atlas在 `447` 个重叠病例中一致 `398` 个（`89.0%`，Cohen's κ `0.838`）。这表明标签具有较高但并非完美的跨数据冻结稳定性，尤其Luminal A/Luminal B边界会受队列、平台与分类版本影响。

## 主要局限性

1. 所有稳健性实验仍来自TCGA-BRCA内部development病例，不等同于独立外部验证，也不能证明跨平台、跨中心或真实临床部署的可迁移性。
2. 替代方案使用冻结的最终随机森林参数而不重新调参。这样能隔离预处理、特征空间和权重变化，但可能低估每个替代方案单独优化后的最佳性能。
3. log2(CPM+1)是透明的count-based library-size normalization敏感性方案，但不替代TMM、DESeq2 VST或跨队列批次校正；TPM与CPM都无法消除肿瘤纯度和细胞组成差异。
4. PAM50标签本身由表达信号定义。使用PAM50基因预测PAM50标签具有概念上的近循环性；排除50个signature基因仍不能移除共表达的代理信号，因此不能把排除结果解释为完全独立于PAM50生物学。
5. Normal-like类别样本较少，五分类结果的不确定性更高；宏平均指标会对该类别的波动较敏感。
6. 当前一病例一样本设计使病例级拆分与样本级拆分都不存在重复病例泄漏。它确认了管线，但无法量化多样本队列中的泄漏幅度。
7. 三个随机种子只能刻画有限的fold/estimator随机性，不能覆盖所有可能的数据划分。
8. TCGA 2012与PanCancer Atlas的标签不一致不能简单视为错误：不同数据冻结、表达平台、可用基因、样本选择及PAM50实现均可能导致边界病例改变亚型。
9. 本阶段不得因任何敏感性结果重新选择模型或再次评估locked test；若要改变最终方案，应在外部队列预先锁定并验证。

## 审计结论

全部场景的训练/验证病例重叠最大值为 `0`，locked-test表达加载与预测均为0。结果支持主结论对所检查分析选择具有总体稳健性，但不能替代外部验证；标签版本差异与PAM50表达定义的近循环性是最重要的解释边界。

图：`figures/01_robustness_performance.png`、`figures/02_random_seed_sensitivity.png`、`figures/03_label_source_concordance.png`。
