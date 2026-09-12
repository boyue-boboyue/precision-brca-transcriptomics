# 协议偏离与澄清记录

## 2026-09-11：单独比较L2与elastic-net

- Git commit：不适用；当前工作目录尚未初始化为Git仓库。
- 修改内容：在通用`multinomial_logistic`候选模型之外，新增独立的L2和elastic-net模型身份；两者固定使用训练折内前1,000个高方差蛋白编码基因。L2搜索`C`，elastic-net搜索`C`与`l1_ratio`。
- 修改原因：直接回答L2 baseline与稀疏可解释elastic-net之间的比较，并使两者共享完全相同的特征数，从而将主要差异限定为正则化方式。
- 时间顺序：`config/logistic_comparison_v1.json`在首次真实队列模型拟合和查看任何模型性能前锁定。修改发生在locked test完全未使用时。
- 可能影响：固定1,000个候选特征没有在本次比较中调节100/500/2,000等特征数，因此结果只支持“1,000个折内高方差基因条件下”的L2与elastic-net比较。后续不同特征数分析应标记为预设敏感性分析。
- 分析类型：development-set主候选模型比较的配置澄清；不改变holdout assignment、outer/inner folds、主要指标或locked-test政策。

运行环境因系统semaphore权限限制，从计划的4进程回退为单进程。该变化只影响运行时间，不改变数据、随机种子、fold、参数网格或评价规则，因此记录为执行环境说明，而非统计协议偏离。

## 2026-09-11：LinearSVC固定1,000个候选特征并进行折内校准

- Git commit：不适用；当前工作目录尚未初始化为Git仓库。
- 修改内容：LinearSVC比较固定使用每个训练折内方差最高的1,000个蛋白编码基因，只搜索`C`；未加入RBF或其他kernel。需要概率时，在每个outer-training partition内部使用预定义的5个inner folds拟合sigmoid `CalibratedClassifierCV`。
- 修改原因：与已完成的L2/elastic-net在相同特征规模下公平比较，同时满足概率不得使用outer-validation或locked-test数据拟合的要求。
- 时间顺序：`config/linear_svc_v1.json`在首次LinearSVC真实队列拟合及查看其性能前锁定。
- 可能影响：本轮结果仅适用于1,000个折内高方差基因；不同特征数量属于后续敏感性分析。未测试RBF，因此不能据此量化非线性kernel的增益或风险。
- 分析类型：development-set主要候选模型比较的配置澄清；不改变holdout assignment、outer/inner folds或主要评价指标。

## 2026-09-11：Random forest的平衡权重与9组锁定候选参数

- Git commit：不适用；当前工作目录尚未初始化为Git仓库。
- 修改内容：新增`config/random_forest_v1.json`，固定`RandomForestClassifier(class_weight="balanced")`并在首次真实拟合前锁定9组候选参数。候选组合共同覆盖500/1,000/2,000个折内高方差基因、250/500/750棵树、无上限/16/32最大深度、1/3/5叶节点最小样本数，以及`sqrt`/0.1/0.2的`max_features`。
- 修改原因：直接回答随机森林比较需求，并在可接受计算量下同时探索模型容量、正则化和特征子采样。
- 时间顺序：配置在首次真实队列随机森林拟合及查看其性能前锁定；189例locked test完全未使用。
- 结果触发说明：五个outer folds均选择2,000个基因这一候选上限。若后续测试更大特征数，必须在运行前另行锁定并标记为查看本轮结果后触发的补充敏感性分析，不能追溯为原始主比较。
- 解释限制：保存的Gini impurity importance仅用于诊断；最终生物学解释必须依赖未参与拟合数据上的permutation importance和跨折稳定性。
- 分析类型：development-set主要候选模型比较的配置澄清；不改变holdout assignment、outer/inner folds、主要指标或locked-test政策。

## 2026-09-11：PAM50 signature基因锁定与排除敏感性

- 修改内容：使用`genefu`官方PAM50 centroid模型文件，固定到commit `9c9b66d1ef22cbfda1df75b626d2dbc68fb9b25c`，将50个signature基因映射到当前表达矩阵。历史符号按当前HGNC/GENCODE轴解析为`CDCA1→NUF2`、`KNTC2→NDC80`、`ORC6L→ORC6`。
- 修改原因：协议要求同时报告PAM50-included与PAM50-excluded性能，以量化使用定义标签的signature基因可能带来的直接信息优势。
- 时间顺序：`config/pam50_exclusion_v1.json`及50基因表在首次excluded模型拟合前锁定；整个分析只使用756例development病例，同一outer/inner folds及原模型网格；locked test使用数为0。
- 实现澄清：50列在低表达过滤、插补、方差筛选、标准化和分类器之前从protein-coding候选轴移除。全部折入选特征与锁定基因零重合。
- 结果：random forest的mean macro F1为included `0.905`、excluded `0.902`；paired fold差为`−0.002 ± 0.010`。balanced accuracy为`0.905`与`0.906`。
- 分析类型：预设敏感性分析；不改变PAM50-included主模型排名。

## 2026-09-11：按预设规则冻结最终模型并进行唯一测试评估

- 修改内容：综合PAM50-included五个候选模型的outer-fold结果，按`config/evaluation.json`预设mean macro F1规则选择random forest。其相对第二名elastic-net的差值为`0.015`，高于`0.01`阈值，未启用特征数/稳定性/解释性平局规则。
- 时间顺序：`config/final_model_lock_v1.json`在加载任何locked-test表达行之前写入并固定了最终模型、完整development内部调参程序、候选参数、随机种子、一次性访问政策和bootstrap方案。
- 测试访问：189例locked test于`2026-09-11T12:17:35Z`开始第1次且唯一一次模型评估；`data/processed/splits/final_test_access_v1.json`记录189行加载与189个预测。不得通过追加模型或超参数重新访问该测试集。
- 结果：balanced accuracy `0.907`（1,000次分层bootstrap 95% CI `0.855–0.945`）；macro F1 `0.885`（`0.836–0.929`）。
- 协议保护：没有为Dummy、L2、elastic-net、LinearSVC或PAM50-excluded候选分别生成locked-test性能，因为这会把测试集用于模型比较并违反一次性评估政策。
- 边界敏感性决定：未在测试解锁前结果驱动地扩展随机森林2,000基因上限。任何更大特征数实验只能作为后续development-only补充分析，不能改变已经完成的locked-test评估。

## 2026-09-13：解释性分析与内部生物学验证

- 修改内容：新增三层解释、稳定基因验证和功能富集。Elastic-net使用已保存外层模型的标准化系数；random forest permutation importance只在对应outer-validation折上计算；TreeSHAP使用100例development-only background解释6例按预测类别、正确性和概率预选的locked-test病例。
- 时间顺序：`config/interpretability_v1.json`在正式permutation、SHAP、稳定基因筛选和富集前锁定。稳定基因规则为任一Elastic-net类别或RF permutation的top 20至少进入3/5折；预期标志基因没有强制入选。
- 生物学验证：所有稳定基因表达和ER/PR/HER2关联仅使用development病例；GO:BP/Reactome主背景为至少一个outer-training低表达过滤器通过的全部蛋白编码基因，5/5折均通过背景作为敏感性分析；多重检验使用BH-FDR。
- 测试访问：解释专用记录只加载6个预锁定病例，计算性能指标数为0，模型选择标志为false；最终189例测试性能没有重算。
- 实现修订：g:Profiler返回的`intersections`是与查询基因逐位对应的证据代码嵌套列表。首次结果序列化在两套原始响应均保存后停止；`config/interpretability_execution_amendment_v1.json`记录了从原始响应重建基因交集的修复及让GO/Reactome分别可见的纯展示调整。查询基因、背景、来源、FDR阈值和原始响应均未改变，也没有重新调用API。
- 解释限制：高相关基因会分摊系数、permutation importance和SHAP值；任何“排名第一”都不是唯一性、因果性、新颖性或临床可用性的证明。

## 2026-09-13：最终模型选择后的稳健性与标签一致性分析

- 修改内容：新增`config/robustness_v1.json`并对已经选择的随机森林执行13个development-only固定模型场景、共65个外层拟合。覆盖四/五分类、PAM50 included/excluded、log2(TPM+1)/log2(CPM+1)、四种低表达规则、三个病例级随机种子、balanced/unweighted、PAM50-only/全蛋白编码转录组，以及病例分组/样本分层拆分；同时比较原始PanCancer Atlas标签链路与TCGA 2012 publication freeze。
- 时间顺序：该分析发生在最终模型选择和唯一一次locked-test评估之后，因此明确属于post-selection敏感性分析。场景、随机种子和固定参数在正式65次拟合前保存；首次5-tree冒烟测试不产生正式结果。正式分析没有加载locked-test表达行、生成测试预测或重新计算测试性能。
- 固定模型理由：沿用最终随机森林的2,000个训练折高方差基因、750棵树、depth 16、leaf 1和`sqrt`特征采样，不对每种替代方案重新调参，以隔离分析选择本身的影响。该设计可能低估替代方案单独优化后的性能。
- 主要结果：四分类参考macro F1为`0.905`；PAM50-excluded `0.899`、log2 CPM `0.911`、三个替代低表达规则`0.906–0.909`、三个病例级随机种子`0.905–0.907`、unweighted `0.878`、PAM50-only `0.917`。五分类macro F1为`0.817`，其中29例Normal-like的pooled F1为`0.512`。
- 标签一致性：原始cBioPortal记录、锁定表、表达样本轴与split表完全一致。TCGA 2012与PanCancer Atlas在447个重叠病例中一致398个（`89.0%`，κ `0.838`）；不同数据冻结、平台和PAM50实现可能导致真实版本差异，不把49个不一致病例自动视作数据错误。
- 解释边界：PAM50-only的较高内部CV性能符合标签由同类表达信号定义的预期，不是临床效用证据；排除50个signature基因也不能移除共表达代理。所有结果只支持内部稳健性，不能替代独立外部验证，也不得据此重新选择模型或再次访问locked test。
- 分析类型：预设项目范围的post-selection development-only敏感性补充及局限性审计；不改变主分析、最终模型锁或已报告的唯一测试集结果。
