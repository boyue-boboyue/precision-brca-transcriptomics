# OncoStratify-BRCA 分析协议

**项目名称：** OncoStratify-BRCA  
**仓库名称：** `precision-brca-transcriptomics`  
**协议版本：** 0.1.3  
**协议日期：** 2026-09-09  
**状态：** 数据、标签、评估配置与拆分已锁定；模型拟合前预设方案（preregistered-style analysis protocol）

## 1. 项目摘要

本项目使用 NCI Genomic Data Commons（GDC）中的 TCGA-BRCA 原发乳腺肿瘤 RNA-seq 数据，开发并评估可解释的多分类机器学习模型，以预测 PAM50 乳腺癌分子亚型。项目将比较多项逻辑回归、线性支持向量机和随机森林，使用病例级分层嵌套交叉验证和独立锁定测试集评估泛化性能，并通过模型系数、置换重要性和 SHAP 分析解释群体及个体预测。

本项目属于回顾性计算研究和方法学展示，不构成临床诊断、预后判断或治疗建议。

## 2. 研究问题与目标

### 2.1 主要研究问题

TCGA-BRCA 原发肿瘤的转录组表达特征能否在严格控制数据泄漏的条件下，准确区分四个主要 PAM50 分子亚型？

主要亚型为：

- Luminal A（LumA）
- Luminal B（LumB）
- HER2-enriched（HER2E）
- Basal-like（Basal）

### 2.2 主要目标

1. 构建可复现的 TCGA-BRCA RNA-seq 病例队列和表达矩阵。
2. 比较多项逻辑回归、线性支持向量机和随机森林的分类性能。
3. 以 balanced accuracy 和 macro F1 评估类别不平衡条件下的性能。
4. 识别稳定、可解释且具有乳腺癌生物学意义的预测特征。
5. 解释模型对独立测试病例作出亚型预测的主要依据。

### 2.3 次要目标

1. 比较包含与排除 PAM50 签名基因时的分类性能。
2. 将 Normal-like 纳入五分类敏感性分析。
3. 检查模型结果在不同预处理参数、随机种子和特征数量下的稳健性。
4. 检查预测结果与 ER、PR、HER2 临床标志物之间的一致性。

## 3. 研究假设

### 3.1 主要假设

至少一个预设机器学习模型在锁定测试集上的 balanced accuracy 和 macro F1 将明显高于仅依据类别先验作出预测的 DummyClassifier。

### 3.2 次要假设

- 允许使用 PAM50 基因的模型将获得最高性能，但该结果主要反映对既有表达型标签的复现。
- 排除 PAM50 基因后，转录组其余部分仍保留可用于亚型区分的信息，但性能预计下降。
- 线性模型能够以更少的特征获得与随机森林相近的性能，并提供更稳定的基因级解释。

## 4. 关键解释边界

PAM50 标签本身来源于基因表达模式，因此使用同一肿瘤的 RNA-seq 特征预测 PAM50 标签存在目标与输入的内生关系。尤其在输入包含 PAM50 签名基因时，本研究评估的是“复现 PAM50 分型的能力”，而不是独立发现新的临床分型工具。

为透明呈现这一问题，所有主要模型必须在以下两个特征方案中分别运行：

1. **PAM50-included：** 允许所有通过质量控制的候选基因进入模型。
2. **PAM50-excluded：** 在任何特征选择前删除 PAM50 签名基因。

论文、README 和图表不得将特征重要性直接解释为因果机制、肿瘤驱动因素或治疗靶点。

## 5. 数据来源与版本记录

### 5.1 RNA-seq 与元数据

- 数据库：NCI Genomic Data Commons
- 项目：TCGA-BRCA
- 数据类别：Transcriptome Profiling
- 数据类型：Gene Expression Quantification
- 实验策略：RNA-Seq
- 工作流：STAR - Counts
- 访问级别：Open
- 样本类型：Primary Tumor

数据获取脚本必须保存：

- 完整 GDC API 请求 JSON；
- 文件 UUID、文件名、MD5 和文件大小；
- case、sample 和 aliquot UUID；
- TCGA case 和 sample barcode；
- GDC data release、API 状态、查询日期和下载日期；
- 原始 manifest 和样本注释文件。

### 5.2 PAM50 标签

主要标签源锁定为 cBioPortal 中 `brca_tcga_pan_can_atlas_2018` 研究的患者级临床属性 `SUBTYPE`。该研究名为 “Breast Invasive Carcinoma (TCGA, PanCancer Atlas)”，标签于2026-09-09通过 cBioPortal REST API 分页获取。原始响应、研究元数据、临床属性定义、获取时间和 SHA-256 均予以保存。

规范化后的锁定标签表为 `data/processed/labels/pancanatlas_pam50_locked.tsv`，锁定清单为 `data/processed/labels/pancanatlas_pam50_lock.json`。锁定表含981个唯一病例，其中945例属于预设四分类，36例为 Normal-like。全部981例均可与当前 GDC STAR Counts 病例精确匹配。

2012年 GDC 论文补充数据仅作为历史一致性检查，不作为主要标签源。不得在查看模型表现后改用其他标签源；如确需改变，必须记录为协议偏离。

不得使用本项目待训练的同一表达矩阵临时计算 PAM50 标签，再将这些标签作为无说明的独立真值。

### 5.3 临床与样本元数据

如数据可用，保留下列字段用于队列描述和一致性检查：

- diagnosis age；
- race 和 ethnicity；
- AJCC stage；
- histological type；
- ER、PR 和 HER2 status；
- tissue source site；
- plate、center 或其他潜在批次字段；
- tumor purity（若有可靠公开来源）。

临床变量默认不作为主要分类器输入，避免模型仅以临床受体状态替代表达信号。

## 6. 队列构建

### 6.1 纳入标准

病例必须同时满足：

1. 属于 TCGA-BRCA。
2. 拥有公开可访问的 STAR - Counts RNA-seq 文件。
3. 样本类型为 Primary Tumor。
4. 能够通过 UUID 或完整 barcode 可靠地映射到唯一病例。
5. 拥有可识别的 PAM50 标签。
6. 表达文件通过完整性和基本质量检查。

### 6.2 排除标准

排除：

- Solid Tissue Normal、Metastatic、Recurrent Tumor 或其他非原发肿瘤样本；
- PAM50 标签缺失、冲突或无法映射的病例；
- 文件损坏、MD5 不匹配或表达矩阵解析失败的样本；
- 预先标记为需排除的 GDC redaction 或关键 annotation 记录；
- 无法确定病例归属的样本；
- 预定义质量控制后仍为显著异常值、且存在明确技术依据的样本。

不得仅因为某个样本在 PCA 中与其 PAM50 标签不一致而排除该样本。

### 6.3 重复样本处理

分析单位为病例，而不是 aliquot。每位病例主分析仅保留一个 Primary Tumor RNA-seq 样本。若存在多个候选样本，按以下固定顺序选择：

1. 排除具有关键 GDC annotation 的样本；
2. 优先选择元数据最完整且文件校验通过的样本；
3. 优先选择测序质量或有效文库规模较高的样本；
4. 若仍并列，按 sample UUID 字典序选择第一个。

最终选择过程必须生成逐病例审计表，不允许人工依据模型结果选择 aliquot。

### 6.4 标签规则

主分析只保留 LumA、LumB、HER2E 和 Basal 四类。Normal-like、无法判定及其他标签从主分析排除，但 Normal-like 将在预设的五分类敏感性分析中重新纳入。

标签名称必须在单独映射表中标准化，原始标签不得覆盖。

### 6.5 队列流程图

必须报告：

- GDC 查询获得的文件数和病例数；
- Primary Tumor 病例数；
- 去除重复样本后的病例数；
- 可匹配 PAM50 标签的病例数；
- 每条排除规则对应的数量；
- 最终四分类和五分类队列中各亚型数量。

## 7. 表达矩阵构建与质量控制

### 7.1 原始矩阵

每个 STAR Counts 文件的列名和文件头必须在解析时验证，禁止仅凭固定列序号读取。主分析使用 GDC 提供的未链特异性 TPM 字段，并构造 `samples × genes` 矩阵。

同时保留 raw counts 矩阵，用于质量控制和可能的补充分析，但不作为主机器学习输入。

截至协议版本0.1.2，病例级矩阵已锁定为1,095个病例 × 60,660个GENCODE v36基因，并保存 `unstranded` counts、`tpm_unstranded` 和 `log2(TPM+1)` 三种表示。全部981个已锁定 PAM50 病例均成功对齐，其中四分类主队列为945例。矩阵构建阶段未执行低表达或方差筛选。

### 7.2 基因注释

- 保存原始 Ensembl gene ID。
- 创建移除版本号后的 Ensembl ID。
- 保存 gene symbol 与 gene biotype。
- 主分析候选特征限定为 protein-coding genes。
- 若多个 Ensembl ID 映射到同一 symbol，模型内部使用 Ensembl ID 保持唯一性；展示结果时同时报告 symbol。

### 7.3 样本级质量控制

在拆分数据前仅允许执行不依赖标签、且不会根据测试分布优化参数的文件完整性检查。记录：

- 总 counts 和非零基因数；
- TPM 总和及表达分布；
- 样本间相关性；
- PCA 中的潜在技术异常；
- 已知 tissue source site、plate 或 center 信息。

任何异常样本排除均须给出技术理由并记录在 `exclusions.tsv`。基于统计阈值的异常处理规则应在查看模型测试性能之前锁定。

## 8. 数据拆分与泄漏控制

### 8.1 锁定测试集

在模型开发前，按 PAM50 亚型分层划分：

- development set：80%；
- locked test set：20%。

若任一类别样本量导致20%测试集无法稳定评估，则可改为85%/15%，但必须在首次模型拟合前记录原因。随机种子默认为 `20260909`，病例 split assignment 必须保存到版本控制中。

锁定测试集在以下事项全部完成前不得用于模型选择：

- 预处理规则锁定；
- 候选模型和超参数空间锁定；
- 主要指标锁定；
- 最终模型选择规则锁定。

截至协议版本0.1.3，拆分已于2026-09-09在首次监督模型拟合前锁定。固定种子为`20260909`：

- 四分类主分析：945例，其中development 756例、locked test 189例；
- 五分类敏感性分析：981例，其中development 785例、locked test 196例；
- 五分类文件完整保留四分类病例的development/test归属，仅另外分配36例Normal-like；
- 当前病例级矩阵没有重复病例，但生成器始终先按`case_barcode`聚合后分配，若同一病例存在多个样本，这些样本必须保持同一归属；病例内标签冲突将直接报错。

权威assignment、配置与SHA-256锁定清单位于`data/processed/splits/`和`config/evaluation.json`。任何会改变现有assignment的重运行均会被脚本拒绝，除非创建新的明确版本并登记协议偏离。

### 8.2 开发集内部验证

使用嵌套分层交叉验证：

- 外层：5-fold StratifiedKFold，评估候选模型；
- 内层：5-fold StratifiedKFold，选择超参数；
- 两层均打乱并使用记录的固定随机种子；
- 若仍存在一位病例多个记录，必须改用病例分组拆分。

截至协议版本0.1.3，development set的外层fold以及每个外层训练集合对应的内层fold均已预先生成并写入逐病例assignment。`outer_fold`表示该病例所属的外层验证折；`inner_fold_outer_K`只对外层K的训练病例赋值，外层验证病例和locked-test病例保持为空。内层随机种子固定为`20260909 + 1000 + outer_fold`。

### 8.3 必须在训练折内部拟合的操作

以下步骤必须实现为 Pipeline 中的 transformer，并仅使用当前训练折拟合：

- 低表达过滤；
- 缺失值处理；
- 方差过滤；
- scaling；
- 单变量或多变量特征选择；
- PCA（若作为模型输入）；
- 类别校准；
- 任何数据驱动的批次校正。

禁止在完整数据集上先筛选差异基因或高变基因，再进行交叉验证。

## 9. 预处理方案

### 9.1 主分析表达变换

主分析输入为：

```text
log2(TPM + 1)
```

### 9.2 低表达过滤

默认在每个训练折内保留满足以下条件的基因：

```text
TPM >= 1 的训练样本比例 >= 10%
```

同一训练折学得的基因集合应用于其验证折或测试集。

### 9.3 缺失值

RNA-seq 表达矩阵不应存在大量随机缺失。若出现缺失，首先检查下载、解析和注释匹配。只有在确认少量且无法恢复后，才允许在训练折内使用中位数插补；插补器不得使用验证或测试数据拟合。

### 9.4 Scaling

- 逻辑回归和线性 SVM：训练折内逐基因标准化为均值0、标准差1。
- 随机森林：主分析不要求 scaling，但必须使用同一训练折内的表达和方差过滤规则。

## 10. 探索性分析

探索性分析包括：

- 各亚型样本数和比例；
- 样本表达分布及质量指标；
- 基于高变基因的 PCA；
- 按 PAM50、批次、tissue source site 和受体状态着色的 PCA；
- 高变基因热图和无监督层次聚类；
- PAM50 与 ER、PR、HER2 的列联图；
- 聚类与 PAM50 的 adjusted Rand index 或 normalized mutual information。

EDA 使用完整队列时必须明确标记为描述性分析。任何从完整队列发现的模式不得随后作为未经嵌套验证的特征选择规则。

## 11. 候选模型

### 11.1 基线模型

- `DummyClassifier(strategy="prior")`
- `DummyClassifier(strategy="stratified")`

### 11.2 多项逻辑回归

比较 L2 与 elastic-net 正则化，使用 multinomial objective 和 `class_weight="balanced"`。预设调参维度包括正则化强度、L1 ratio 和最多保留特征数。

### 11.3 线性支持向量机

使用 LinearSVC 和 `class_weight="balanced"`。主要调节参数为 `C` 和特征数。若 ROC-AUC 或 SHAP 实现需要概率，使用仅在训练数据内部拟合的 `CalibratedClassifierCV`。

### 11.4 随机森林

使用 class-balanced 权重，并在训练折内先进行低表达和方差/特征数量限制。预设调参维度包括：

- `n_estimators`；
- `max_depth`；
- `min_samples_leaf`；
- `max_features`；
- 候选特征数量。

### 11.5 类别不平衡

主分析使用类别权重，不使用 SMOTE。高维表达空间中的合成样本可能缺少生物学合理性；如后续探索 SMOTE，必须标记为补充分析，且仅在训练折内执行。

## 12. 超参数选择与最终模型规则

所有模型的超参数搜索空间必须在首次查看锁定测试集前保存至配置文件。内层交叉验证以 mean macro F1 最大化为选择标准。

预处理、候选模型、超参数网格、评价指标和并列模型裁决规则已锁定在`config/evaluation.json`。开发集运行器`scripts/run_nested_cv.py`只能读取development病例进行拟合和预测；locked-test病例不进入表达矩阵切片。

候选模型的主排名依据外层交叉验证 mean macro F1；balanced accuracy 作为共同主要指标报告。若两个模型的 mean macro F1 差异小于0.01，则优先选择：

1. 特征更少的模型；
2. 结果更稳定的模型；
3. 更易解释的模型；
4. 计算复杂度更低的模型。

选定模型后，使用完整 development set 重新执行内部调参和拟合，并仅对 locked test set 评估一次。

## 13. 评价指标

### 13.1 主要指标

- Macro F1-score
- Balanced accuracy

### 13.2 次要指标

- Overall accuracy
- 每类 precision、recall、specificity 和 F1
- Macro 和 weighted one-vs-rest ROC-AUC
- 每类 ROC-AUC
- Macro average precision 和每类 PR-AUC
- Multiclass log loss（仅对经过适当校准的概率模型）
- Confusion matrix（原始计数及按真实类别归一化）

### 13.3 不确定性

- 报告嵌套交叉验证外层折的均值、标准差和完整分布。
- 对 locked test set 指标执行病例级分层 bootstrap，默认1,000次。
- 若某次 bootstrap 缺少一个类别，则丢弃该次并报告有效重复次数。
- 置信区间使用 bootstrap percentile 95% interval。

测试集置信区间用于描述不确定性，不用于反复选择模型。

## 14. 模型解释

### 14.1 全局解释

- 逻辑回归：报告标准化输入下每一类别的系数方向和绝对值。
- 线性 SVM：报告每类决策函数权重。
- 所有最终模型：在 held-out 数据上计算 permutation importance。
- 随机森林：可补充 impurity-based importance，但不得将其作为唯一重要性依据。

### 14.2 稳定性分析

在嵌套交叉验证的每个外层训练折中记录 top features，报告：

- 进入 top 20、top 50 和 top 100 的频率；
- 系数或重要性方向的一致性；
- 不同随机种子下的排名相关性；
- PAM50-included 与 PAM50-excluded 方案中的差异。

### 14.3 个体解释

根据模型类型使用适当 SHAP explainer：

- 线性模型使用 linear explainer；
- 随机森林使用 tree explainer。

SHAP background 数据只能来自 development/training data。展示若干预先定义的 locked test 病例，包括：

- 每类至少一个正确且高置信预测；
- 至少一个错误预测；
- 至少一个低置信或类别边界病例。

病例选择规则必须基于预测类别、正确性和置信度，不得仅挑选最符合生物学叙事的样本。

## 15. 生物学解释与富集分析

对跨折稳定的重要基因执行：

1. 各亚型表达分布检查；
2. 与 PAM50 基因集重叠分析；
3. GO Biological Process、Reactome 或 MSigDB Hallmark 富集；
4. 与 ER、PR、HER2 状态的一致性分析；
5. 已知乳腺癌标志物的文献对照。

富集分析背景集必须定义为相应实验中通过基础表达过滤、因而有机会被模型选择的所有基因。若同时测试多个通路，使用 Benjamini-Hochberg 方法控制 FDR，默认显著性阈值为 `q < 0.05`。

## 16. 预设敏感性分析

至少完成以下分析：

1. 四分类与包含 Normal-like 的五分类比较。
2. PAM50-included 与 PAM50-excluded 比较。
3. 仅使用 PAM50 基因与使用全转录组比较。
4. 低表达阈值10%与20%比较。
5. 不同候选特征数量比较。
6. 类别权重与不加权模型比较。
7. 至少三个预设随机种子下重复主要模型流程。
8. 如数据允许，检查组织来源中心或技术批次对性能的影响。

敏感性分析不得替代预设主分析。若敏感性结果与主分析冲突，应并列报告并讨论原因。

## 17. 偏差与公平性检查

在样本量允许且不会泄露受控信息时，按以下变量描述性能：

- age group；
- race；
- ethnicity；
- stage；
- histological type。

只有当子组测试病例数足够时才计算指标；建议每个子组至少20例且每个目标类别至少5例。样本不足时仅报告分布，不作稳定的性能结论。

## 18. 可复现性要求

项目必须满足：

- Python 和所有依赖版本锁定；
- 所有随机过程设置并记录 seed；
- 原始数据只读，衍生数据由脚本生成；
- manifest、MD5、API 查询和排除日志纳入版本控制；
- notebook 只用于叙事和展示，核心逻辑位于可测试模块；
- 每个主要表格和图片均能由命令重新生成；
- 自动化测试覆盖 barcode 映射、重复病例处理、矩阵维度和防泄漏 transformer；
- 运行日志记录 Git commit、配置文件和执行时间。

评估框架另外要求：

- `scripts/verify_evaluation_framework.py`验证病例唯一性、集合隔离、五层外/内折完整性、分层覆盖、锁定文件哈希和测试集未访问标志；
- 单元测试确认低表达与方差选择的support只由训练数据学习，并验证重复样本不会跨patient fold；
- 所有监督候选模型必须通过同一Pipeline执行`low-expression filter → median imputation → variance selection → scaling → classifier`。

预期命令接口：

```bash
make metadata
make download
make cohort
make matrix
make eda
make train
make evaluate
make explain
make test
```

## 19. 预期输出

### 19.1 数据与审计文件

- GDC manifest 和查询 JSON
- `cohort.tsv`
- `exclusions.tsv`
- `split_assignments.tsv`
- 处理后的表达矩阵及特征元数据
- 数据版本和校验和报告

### 19.2 表格

- 队列特征和类别分布
- 嵌套交叉验证模型比较
- 锁定测试集性能及95%置信区间
- 每类评价指标
- 稳定重要基因列表
- 敏感性分析汇总

### 19.3 图形

- 队列纳入排除流程图
- 类别分布图
- PCA 和聚类热图
- 混淆矩阵
- one-vs-rest ROC 和 PR 曲线
- 模型性能比较图
- 全局特征重要性图
- 代表性个体 SHAP 图
- 重要基因的亚型表达图
- 通路富集图

## 20. 局限性

研究结论必须明确考虑：

- TCGA 是回顾性、单队列数据，不能替代前瞻性临床验证；
- PAM50 标签由表达信号产生，存在标签—特征内生性；
- 各亚型样本量不平衡，尤其是 HER2E 和 Normal-like；
- 肿瘤纯度、基质、免疫浸润和批次因素可能影响表达；
- TCGA 的人群构成和诊疗年代可能限制外部适用性；
- 高维相关特征会使单基因重要性不稳定；
- 准确复现分子亚型不等于改善患者结局或治疗选择。

如未来增加独立队列，外部验证必须冻结现有预处理器、特征集合、模型参数和标签映射后进行。

## 21. 协议偏离管理

首次模型拟合后，对本协议的任何修改都必须追加到 `docs/deviations.md`，记录：

- 日期和 Git commit；
- 修改内容；
- 修改原因；
- 修改发生在查看哪些结果之前或之后；
- 对主要结论可能造成的影响；
- 该分析属于主分析、敏感性分析还是探索性分析。

不得删除或覆盖旧协议版本。重大修改应递增协议版本并创建 Git tag。

## 22. 完成标准

只有在以下条件全部满足后，项目才视为完成：

1. 可从保存的 GDC manifest 重建分析队列。
2. 每个纳入和排除病例均可追溯。
3. 所有预处理均在交叉验证折内执行并有测试验证。
4. 三类候选模型及 DummyClassifier 均完成嵌套验证。
5. locked test set 仅在模型与规则锁定后评估。
6. PAM50-included、PAM50-excluded 和五分类敏感性分析均完成。
7. 性能、解释性、稳定性和局限性结果均被报告。
8. 从干净环境可通过文档化命令重新生成主要结果。

## 23. 参考资料

- [NCI Genomic Data Commons: TCGA-BRCA](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)
- [GDC mRNA Analysis Pipeline](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/)
- [GDC API Search and Retrieval](https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/)
- [GDC File Downloading](https://docs.gdc.cancer.gov/API/Users_Guide/Downloading_Files/)
- [TCGA: Comprehensive molecular portraits of human breast tumours](https://gdc.cancer.gov/about-data/publications/brca_2012)
- [UCSC Xena PAM50 tutorial](https://ucsc-xena.gitbook.io/project/tutorials/advanced-tutorial-section-2)
