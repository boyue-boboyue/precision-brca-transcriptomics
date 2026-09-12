# OncoStratify-BRCA 项目交接说明

**快照日期：** 2026-09-13
**工作目录：** `oncostratify-brca-interpretable-machine-learning-for`  
**协议版本：** 0.1.3  
**当前阶段：** 数据、标签、表达矩阵、探索性分析、严格评估框架、五模型nested CV、PAM50排除敏感性、最终模型冻结、唯一一次locked-test评估、三层模型解释及内部生物学验证均已完成；性能与解释验证状态均为`PASS`。

## 1. 可从这里继续

下一步科学重点应为独立外部队列验证：冻结现有预处理规则、2,000基因选择程序、随机森林参数和PAM50标签映射后，在新队列上检验性能、特征稳定性及解释迁移性。189例locked test已按冻结规则评估一次；解释阶段只读取预先按预测表现选定的6例并没有重算性能，禁止再用测试集进行候选模型比较、追加调参或模型选择。

随机森林五折均命中特征数网格上限。若探索高于2,000个基因，必须标记为查看结果后触发的development-only补充敏感性分析，且不得借此重新访问locked test。

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

### 2.8 PAM50 signature 排除敏感性

- 使用`genefu`官方PAM50 centroid文件，固定到commit `9c9b66d1ef22cbfda1df75b626d2dbc68fb9b25c`，锁定50个基因及当前矩阵列。
- 历史符号解析：`CDCA1→NUF2`、`KNTC2→NDC80`、`ORC6L→ORC6`；50个基因均唯一匹配protein-coding矩阵列。
- 排除发生在低表达过滤、插补、方差筛选、标准化和分类器之前；所有模型沿用原网格及同一5×5 folds。
- excluded mean macro F1：Dummy `0.173`、L2 `0.871`、elastic-net `0.866`、LinearSVC `0.871`、random forest `0.902`。
- random forest 的excluded − included配对折差：macro F1 `−0.002 ± 0.010`；balanced accuracy `+0.001 ± 0.014`。
- 每个模型有756个唯一OOF预测；全部折所选特征与锁定PAM50基因零重合；locked-test使用数为0。

### 2.9 最终模型与统一性能报告

- 主分析按预设mean outer-fold macro F1排序。random forest `0.905`，elastic-net `0.890`，差值`0.015`高于`0.01`平局阈值，因此未触发简约性平局规则。
- 最终模型锁：PAM50-included balanced random forest；完整development内部调参仍选择2,000基因、750棵树、depth 16、leaf 1、`max_features="sqrt"`。
- 189例locked test仅评估一次：balanced accuracy `0.907`（95% CI `0.855–0.945`），macro F1 `0.885`（`0.836–0.929`），accuracy `0.884`，macro OvR ROC-AUC `0.981`，macro average precision `0.927`。
- 测试集逐类F1：Luminal A `0.899`、Luminal B `0.773`、Basal-like `0.985`、HER2-enriched `0.882`。
- 小类别PR-AUC/AP：Basal-like `0.994`（n=34），HER2-enriched `0.887`（n=16）。
- 置信区间使用1,000次病例级、按真实类别分层的percentile bootstrap；有效重复数1,000。
- 统一报告包含两种特征方案、五个模型的逐折均值/SD、聚合OOF混淆矩阵、逐类precision/recall/F1、ROC-AUC、PR-AUC、曲线点及最终测试结果；验证状态`PASS`。

### 2.10 三层解释与生物学验证

- Elastic-net：从已保存的五个外层模型汇总标准化系数；未选择基因按0计入跨折均值，保存每类top positive/negative及top 20/50/100指示。稳定性主指标为各基因在5个外层折进入top 20的次数。
- Random forest：每折仅在outer-train拟合，在相应outer-validation上执行2,000特征×5次重复置换，同时记录macro F1和balanced accuracy下降；permutation阶段加载locked-test表达行数为0。
- 全局held-out permutation前列包括`CENPF`、`FAM83D`、`CENPA`、`ASPM`、`FOXM1`、`KIF20A`和`TOP2A`，但数值较小且受相关特征分摊影响，不能把第一名解释为唯一关键基因。
- 类别级Elastic-net信号包括Luminal B的`ESR1` positive、Basal-like的`FOXC1` positive和`FOXA1` negative、HER2-enriched的`ERBB2/GRB7` positive；所有结果均来自数据，没有强制插入预期标志物。
- SHAP：100例background全部来自development并在解释前锁定；解释6例prediction-defined测试病例（四类各1例正确高置信、1例高置信误分、1例最低概率间隔病例）。四类×2,000特征的概率归因最大加和残差`4.02e-09`。
- 稳定基因规则在查看解释结果前固定为“任一Elastic-net类别或RF permutation的top 20至少出现3/5折”，得到58个基因；14个与锁定PAM50 50基因重叠，4个属于预声明检查基因（`ESR1`、`ERBB2`、`FOXA1`、`KRT17`）。
- 稳定基因的逐development病例表达、四亚型分布、Kruskal-Wallis+BH、PAM50重叠、ER/PR/HER2 Positive-vs-Negative Mann-Whitney+BH及rank-biserial效应量均已保存。
- 证据分层：14个`known_marker_or_PAM50`、34个`clinically_correlated_non_PAM50`、10个`potential_candidate_not_novelty_claim`；后者只是需要外部与功能验证的候选，不是新颖性或因果声明。
- g:Profiler GO:BP/Reactome使用15,238个至少一折通过训练折低表达过滤的蛋白编码基因作为主背景，15,118个5/5折通过基因作为敏感性背景，BH-FDR 0.05。主结果包括estrogen response、ERBB/EGFR signaling及Reactome `GRB7 events in ERBB2 signaling`。
- 解释与生物学验证端到端校验状态为`PASS`；包括分区隔离、background来源、病例选择、SHAP加和、PAM50重叠、富集背景、受体多重检验、输出哈希和65张图形。

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
- `scripts/lock_pam50_signature.py`：把50个PAM50 signature基因锁定到当前表达矩阵列。
- `scripts/run_pam50_excluded_nested_cv.py`：在Pipeline前排除PAM50基因并重跑全部五个模型。
- `scripts/lock_final_model.py`：仅使用development外层结果应用预设规则并冻结最终模型。
- `scripts/run_final_locked_test.py`：完整development内部调参、一次性locked-test评估及分层bootstrap。
- `scripts/generate_unified_performance_report.py`：从已保存预测生成统一表格、ROC/PR曲线和报告。
- `scripts/verify_unified_performance_report.py`：验证OOF完整性、PAM50零重合、模型锁、测试集唯一访问、bootstrap和产物哈希。
- `scripts/lock_interpretability_plan.py`：在解释拟合前锁定病例、SHAP background、置换次数、稳定阈值和生物学验证规则。
- `scripts/run_interpretability.py`：Elastic-net系数稳定性、outer-validation permutation importance及development-background TreeSHAP。
- `scripts/run_biological_validation.py`：稳定基因表达分布、PAM50重叠、IHC受体关联与证据分层。
- `scripts/fetch_functional_enrichment.py`：使用表达过滤背景请求并锁定g:Profiler GO:BP/Reactome结果。
- `scripts/generate_interpretability_report.py`、`verify_interpretability.py`：生成解释/生物学报告、65张图并进行端到端验证。

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
- `config/pam50_exclusion_v1.json`与`data/processed/labels/pam50_signature_genes_v1.tsv`：PAM50排除方案及50基因锁。
- `outputs/modeling/pam50_excluded/`：五模型excluded nested-CV指标、OOF预测、参数与入选特征审计。
- `config/final_model_lock_v1.json`：最终模型、主排名、阈值裁决和唯一测试评估程序锁。
- `data/processed/splits/final_test_access_v1.json`：不可追加第二次评估的一次性访问记录。
- `outputs/final_evaluation/`：最终模型调参、189例预测、逐类结果、curve points和1,000次bootstrap。
- `outputs/performance_report/`：统一Markdown报告、TSV表、9张图、manifest和`PASS`验证报告。
- `config/interpretability_v1.json`：解释病例、development-only background、稳定性和生物学验证的预运行锁。
- `config/interpretability_execution_amendment_v1.json`：g:Profiler嵌套交集字段序列化修复及纯展示层修订；统计方案、查询、背景和阈值均未改变。
- `data/processed/interpretability/`：6例SHAP病例及100例development background的锁定表。
- `data/processed/splits/interpretation_test_access_v1.json`：解释专用测试访问记录；加载6行、性能指标0、模型选择false。
- `outputs/interpretability/`：三层解释表、稳定性、完整SHAP、内部生物学验证、富集原始响应、综合报告、65张图与`PASS`验证报告。

## 4. 恢复环境与校验

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-interpretability.txt
.venv/bin/python scripts/verify_expression_matrix.py
.venv/bin/python scripts/verify_eda.py
.venv/bin/python scripts/verify_evaluation_framework.py
.venv/bin/python scripts/verify_logistic_comparison.py
.venv/bin/python scripts/verify_linear_svc.py
.venv/bin/python scripts/verify_random_forest.py
.venv/bin/python scripts/verify_unified_performance_report.py
.venv/bin/python scripts/verify_interpretability.py
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

1. 使用冻结预处理、标签映射、2,000基因程序和模型参数进行独立外部队列验证。
2. 在外部队列检查58个稳定基因的方向、排序、受体关联和SHAP解释是否迁移。
3. 对高度相关基因做基因模块/通路层解释，避免把单基因排名误认为唯一机制。
4. 对10个`potential_candidate_not_novelty_claim`候选开展独立文献与功能验证；不得据内部关联直接声称新颖性或临床效用。
5. 如探索高于2,000个基因，先另行锁定development-only补充方案，且不得重新评估locked test。

## 7. 注意事项

- 项目已初始化为Git仓库并推送到私有GitHub仓库`boyue-boboyue/precision-brca-transcriptomics`。
- 原始`split_lock.json`保留为首次最终评估前的不可变证书；最终访问状态记录在追加式`final_test_access_v1.json`，不得删除或重写以尝试第二次测试评估。
- `.venv`、原始STAR Counts和大型矩阵不应直接提交到普通Git仓库；建议使用数据版本库、对象存储或Git LFS。
- 全队列EDA只用于描述，不能把观察到的HVG、PCA或cluster结果当作未经交叉验证的特征选择依据。
- PAM50标签来自表达模式，因此本项目评估的是对既有PAM50分型的复现能力，而不是独立发现新的临床亚型。
