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
