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
