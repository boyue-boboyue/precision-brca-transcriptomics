# OncoStratify-BRCA 项目交接说明

**快照日期：** 2026-09-11  
**工作目录：** `oncostratify-brca-interpretable-machine-learning-for`  
**协议版本：** 0.1.3  
**当前阶段：** 数据、标签、表达矩阵、探索性分析和严格评估框架已完成；Dummy、multinomial logistic L2、elastic-net、LinearSVC及balanced random forest的development-only nested CV已完成，locked test尚未评估。

## 1. 可从这里继续

下一步应综合已完成候选模型，并把随机森林五折均命中特征数网格上限这一发现作为明确标记的补充敏感性分析；同时完成预设的PAM50基因排除敏感性分析。由于上限问题是在查看结果后发现，扩展特征数不得追溯称为原始主分析。完成后冻结最终模型、超参数选择流程和决策规则。

189例四分类locked test病例不得用于预处理拟合、特征选择、模型比较或性能估计。只有development-set模型选择和最终模型规则冻结后，才允许进行一次最终测试评估。

不要根据完整队列的 EDA 结果手工挑选预测基因。低表达过滤、方差筛选、特征选择、缺失值处理和标准化必须在每个训练折内部拟合。

## 2. 已完成工作

### 2.1 数据获取与锁定

- GDC TCGA-BRCA STAR Counts：1,111个 Primary Tumor 文件，覆盖1,095个唯一病例。
- 下载体积约4.390 GiB，文件大小和MD5校验均通过。
- GDC病例、临床、样本、aliquot和annotation元数据已保存。
- PanCancer Atlas PAM50 主标签源：cBioPortal `brca_tcga_pan_can_atlas_2018` 的患者级 `SUBTYPE`。
- 标签获取时间：2026-09-09T06:18:07Z；原始响应 SHA-256：`a5f86a68e58f9e0db55f0ee826ddb87a83f3a361600b8add598e87cdc919c09f`。
- 锁定标签表 SHA-256：`d228bac88aa323cb510510cdb0f787f5b2e78dfb56b573e4f26715c41d1c7ce8`。
- 标签计数：Luminal A 499、Luminal B 197、Basal-like 171、HER2-enriched 78、Normal-like 36。
- ER/PR/HER2 临床状态来自 cBioPortal `brca_tcga`，使用 TCGA patient barcode 匹配；来源文件和SHA-256已锁定。

### 2.2 表达矩阵

- 形状：1,095样本 × 60,660基因。
- 蛋白编码基因：19,962个。
- `counts_uint32.npy`：GDC unstranded raw counts。
- `tpm_float32.npy`：GDC TPM。
- `log2_tpm_float32.npy`：log2(TPM + 1)。
- 五分类标签样本：981个。
- 四分类主分析样本：945个。
- 一个病例因关键GDC注释标记为不适合分析；该病例本身没有锁定PAM50标签，因此不改变981/945的建模队列数。
- 三个主矩阵SHA-256：
  - counts：`8a820d8dd77a47ec8b133d7221bd9d36a95a7b338c49d1ab97589a9b06f10eb4`
  - TPM：`6e7a19af690dccb941bebf06acccc1429861918417625a39f8e4ab5fee1273c8`
  - log2 TPM：`992cf12149d7e2bf1e0185a90560835bc9643e89397c2a1200ee4d192eef3542`

### 2.3 探索性分析

EDA已生成并验证以下10类图表：

1. 样本纳入排除流程图；
2. 四类与五类 class balance；
3. 逐样本蛋白编码基因表达分布与文库规模；
4. PAM50着色PCA；
5. TSS、RNA plate和legacy RNA-seq batch着色PCA；
6. 前500个高变蛋白编码基因层次聚类热图；
7. k=5层次聚类与PAM50对应关系；
8. ER、PR、HER2与PAM50列联图；
9. subtype在技术/采集标签中的组成；
10. 批次混杂效应量和批次单变量预测能力。

主要描述性结果：

- PC1解释18.14%，PC2解释8.23%，PC1–PC10累计解释46.29%变异。
- 前500 HVG的层次聚类与PAM50对应：ARI 0.296，AMI 0.408，cluster purity 0.680。
- RNA plate与subtype的校正 Cramer's V 为0.117，2,000次置换 p=0.0125；plate-only五折 balanced accuracy为0.288，相比五分类机会水平0.20仅显示有限但非零重合。
- TSS的校正 Cramer's V 为0.113，置换 p=0.0250；legacy batch的 V 为0.080，置换 p=0.1004。
- Aliquot center code和GDC source center在当前RNA队列中均为常数，无法用于批次比较。
- 批次结果不支持近乎完全的技术标签泄漏，但后续应补充按plate/TSS分组的敏感性验证。

### 2.4 严格评估框架

- 固定随机种子：`20260909`。
- 四分类主分析：756例development、189例locked test。
- 五分类敏感性分析：785例development、196例locked test；四分类病例的holdout归属保持不变。
- development内部预先锁定5-fold外层与每个外层训练集中的5-fold内层分层fold。
- 当前矩阵每病例一个样本；拆分生成器仍按`case_barcode`分组，并已用重复样本合成测试验证不会发生patient leakage。
- Pipeline顺序固定为低表达过滤、中位数插补、训练折方差选择、标准化、分类器。
- 低表达、插补、特征选择和标准化均仅在当前内层训练折拟合。
- `outputs/evaluation/verification_report.json`状态为`PASS`；建立框架时未拟合监督模型，也未评估locked test。

### 2.5 Dummy与multinomial logistic比较

- 队列：756例四分类development病例；5-fold outer × 5-fold inner stratified CV。
- Dummy prior：macro F1 `0.173 ± 0.001`，balanced accuracy `0.250 ± 0.000`。
- L2 logistic：macro F1 `0.866 ± 0.043`，balanced accuracy `0.881 ± 0.042`，macro OvR ROC-AUC `0.981 ± 0.008`。
- Elastic-net logistic：macro F1 `0.890 ± 0.053`，balanced accuracy `0.901 ± 0.050`，macro OvR ROC-AUC `0.985 ± 0.006`。
- Elastic-net相对L2的配对外层fold macro F1平均差为`+0.024`，5折中4折更高。
- Elastic-net平均保留170.8/1,000个非零基因，范围128–252；L2的1,000个基因均有非零系数。
- 两种logistic模型均使用`class_weight="balanced"`；没有出现收敛警告。
- 独立验证状态`PASS`：每个模型均产生756个唯一development OOF预测，使用locked-test病例数为0。

### 2.6 LinearSVC与训练数据内概率校准

- 分类器：`LinearSVC(class_weight="balanced")`，固定linear decision function；RBF模型拟合数为0。
- 内层搜索：`C ∈ {0.001, 0.01, 0.1, 1, 10}`；4个外层折选择`C=0.001`，1折选择`C=0.01`。
- 概率：每个outer-training partition内部使用相同的5个预定义inner folds拟合`CalibratedClassifierCV(method="sigmoid", ensemble=True)`。
- 原始LinearSVC：macro F1 `0.872 ± 0.034`、balanced accuracy `0.890 ± 0.042`、accuracy `0.880 ± 0.024`。
- 校准概率：macro OvR ROC-AUC `0.977 ± 0.010`、multiclass log-loss `0.321 ± 0.051`。
- 相对L2 logistic的macro F1平均差`+0.006`；相对elastic-net logistic为`−0.018`，仅2/5折高于elastic-net。
- LinearSVC的1,000个候选基因均有非零权重，不及elastic-net稀疏。
- 无收敛警告；独立验证状态`PASS`；使用locked-test病例数为0。

### 2.7 Balanced random forest

- 分类器：`RandomForestClassifier(class_weight="balanced")`；低表达过滤、高方差筛选、中位数插补和标准化均在当前训练折内拟合。
- 调参覆盖树数量、最大深度、叶节点最小样本数、`max_features`和折内高方差基因数；首次真实拟合前锁定9组候选参数。
- 5×5 nested CV：macro F1 `0.905 ± 0.038`、balanced accuracy `0.905 ± 0.041`、accuracy `0.917 ± 0.020`、macro OvR ROC-AUC `0.986 ± 0.005`、log-loss `0.367 ± 0.016`。
- 五个外层折均选择：2,000个高方差基因、750棵树、`max_depth=16`、`min_samples_leaf=1`、`max_features="sqrt"`。
- 相对elastic-net的同折macro F1平均差为`+0.015`，但仅2/5折更高；balanced accuracy平均差仅`+0.004`，不能据均值宣称稳定优势。
- 相对LinearSVC的同折macro F1平均差为`+0.033`，5/5折更高；相对L2 logistic为`+0.039`，4/5折更高。
- OOF逐类F1：Luminal A `0.936`、Luminal B `0.826`、Basal-like `0.989`、HER2-enriched `0.874`。
- 随机森林具有当前最高macro F1点估计，但log-loss劣于elastic-net（`0.367` vs `0.268`），且约2,000个基因获得非零impurity importance；elastic-net平均仅171个非零基因，更稀疏、解释性更强。
- Gini impurity importance仅保留作诊断，不能直接作为生物学结论；最终解释仍需使用未参与拟合数据上的permutation importance并检查跨折稳定性。
- 独立验证状态`PASS`；产生756个唯一development OOF预测；使用locked-test病例数为0。

## 3. 关键文件

### 协议与报告

- `docs/protocol.md`：正式分析协议。
- `outputs/eda/eda_report.md`：EDA方法、结果和解释边界。
- `outputs/eda/eda_summary.json`：机器可读EDA摘要。
- `outputs/eda/artifact_manifest.json`：EDA文件哈希。
- `outputs/eda/verification_report.json`：EDA校验报告，状态应为 `PASS`。

### 表达矩阵与标签

- `data/processed/expression/matrix_manifest.json`：矩阵来源、选择规则、维度和哈希。
- `data/processed/expression/samples.tsv`：矩阵行轴、病例、样本、aliquot及PAM50标签。
- `data/processed/expression/genes.tsv`：矩阵列轴及基因注释。
- `data/processed/expression/pam50_four_class_eligible_sample_indices.npy`：945个主分析样本索引。
- `data/processed/labels/pancanatlas_pam50_lock.json`：标签锁定信息。

### 复现脚本

- `scripts/query_gdc.sh`、`download_gdc_star_counts.sh`、`verify_gdc_download.sh`：GDC数据获取与校验。
- `scripts/fetch_pancanatlas_pam50.sh`、`lock_pancanatlas_pam50.py`：PAM50标签获取与锁定。
- `scripts/build_expression_matrix.py`、`verify_expression_matrix.py`：表达矩阵构建与校验。
- `scripts/fetch_cbioportal_receptors.sh`：ER/PR/HER2数据获取与锁定。
- `scripts/run_eda.py`、`verify_eda.py`：EDA生成与校验。
- `scripts/lock_evaluation_splits.py`：生成并锁定病例级holdout及嵌套fold assignment。
- `scripts/evaluation_framework.py`：防泄漏transformer、Pipeline和预定义fold读取接口。
- `scripts/run_nested_cv.py`：只使用development病例的嵌套交叉验证运行器。
- `scripts/verify_evaluation_framework.py`：拆分、哈希和防泄漏测试总校验。
- `scripts/run_logistic_comparison.py`：Dummy、L2和elastic-net的development-only 5×5 nested CV。
- `scripts/summarize_logistic_comparison.py`：汇总逐类指标、混淆矩阵和模型比较报告。
- `scripts/verify_logistic_comparison.py`：验证逐病例预测、正则化设置、系数轴、哈希和locked-test隔离。
- `scripts/run_linear_svc.py`：LinearSVC nested CV及outer-training内部的概率校准。
- `scripts/summarize_linear_svc.py`：汇总逐类结果、混淆矩阵及与逻辑回归的同折比较。
- `scripts/verify_linear_svc.py`：验证linear-only约束、校准范围、逐病例概率、系数与locked-test隔离。
- `scripts/run_random_forest.py`：balanced random forest的development-only nested CV及折内高方差筛选。
- `scripts/summarize_random_forest.py`：汇总逐类结果、混淆矩阵及与全部已完成模型的同折比较。
- `scripts/verify_random_forest.py`：验证参数锁、逐病例概率、特征重要性轴、报告哈希和locked-test隔离。

### 评估配置与输出

- `config/evaluation.json`：种子、拆分、Pipeline、候选模型、超参数和选择规则锁定文件。
- `data/processed/splits/four_class_split_assignments.tsv`：四分类逐病例assignment。
- `data/processed/splits/five_class_split_assignments.tsv`：五分类逐病例assignment。
- `data/processed/splits/split_lock.json`：输入、配置和assignment的SHA-256锁定清单。
- `outputs/evaluation/`：便于审阅的assignment副本、类别分布、框架说明与验证报告。
- `config/logistic_comparison_v1.json`：首次logistic比较前锁定的模型与超参数配置。
- `outputs/modeling/logistic_comparison/`：逐折指标、OOF预测、最佳参数、系数、逐类结果、报告和验证清单。
- `config/linear_svc_v1.json`：首次LinearSVC拟合前锁定的C网格、linear-only约束和校准方案。
- `outputs/modeling/linear_svc/`：LinearSVC逐折指标、OOF决策值与校准概率、参数、系数、比较报告及验证清单。
- `config/random_forest_v1.json`：首次随机森林拟合前锁定的9组参数、balanced类别权重和特征筛选规则。
- `outputs/modeling/random_forest/`：随机森林逐折指标、OOF概率、最佳参数、impurity importance、全模型比较、报告及验证清单。

## 4. 恢复环境与校验

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-eda.txt
.venv/bin/python scripts/verify_expression_matrix.py
.venv/bin/python scripts/verify_eda.py
.venv/bin/python scripts/verify_evaluation_framework.py
.venv/bin/python scripts/verify_logistic_comparison.py
.venv/bin/python scripts/verify_linear_svc.py
.venv/bin/python scripts/verify_random_forest.py
```

预期所有校验脚本均输出 `"status": "PASS"`。

## 5. 导出包说明

导出目录为 `outputs/exports/`：

- `oncostratify-brca-core-2026-09-09.tar.gz`：代码、协议、元数据、锁定标签、矩阵轴与索引、EDA图表和结果；不含三个大型表达矩阵，也不含4.4 GiB原始STAR Counts。
- `oncostratify-brca-processed-data-2026-09-09.tar.gz`：三个大型表达矩阵及其轴、索引、标签、校验文件和恢复所需脚本。解压此包和core包即可继续建模，无需重新下载原始GDC文件。
- `archive_checksums.sha256`：两个归档文件的SHA-256。
- `export_summary.json`：归档大小、成员数、校验和及恢复顺序。

原始GDC STAR Counts没有进入导出包；若要从原始数据完全重建，使用 `data/manifests/gdc_manifest_tcga_brca_star_counts.tsv` 和现有下载脚本重新获取。

## 6. 下一阶段建议清单

1. 将随机森林特征数高于2,000的开发集分析登记为结果触发的补充敏感性分析，并在运行前锁定范围和判断标准。
2. 分别运行PAM50-included和PAM50-excluded特征方案；后者仍需补充并锁定PAM50基因清单。
3. 综合全部候选模型的外层fold表现、折间稳定性、概率质量、稀疏性与可解释性。
4. 按预设macro F1规则和明确的平局/简约性规则确定最终模型并冻结最终配置。
5. 只在上述步骤完成后评估一次locked test set。
6. 随后进行系数、held-out permutation importance、SHAP和跨折特征稳定性分析。

## 7. 注意事项

- 当前目录不是Git仓库；若后续需要版本控制，应在开始监督学习前初始化并提交当前快照。
- `.venv`、原始STAR Counts和大型矩阵不应直接提交到普通Git仓库；建议使用数据版本库、对象存储或Git LFS。
- 全队列EDA只用于描述，不能把观察到的HVG、PCA或cluster结果当作未经交叉验证的特征选择依据。
- PAM50标签来自表达模式，因此本项目评估的是对既有PAM50分型的复现能力，而不是独立发现新的临床亚型。
