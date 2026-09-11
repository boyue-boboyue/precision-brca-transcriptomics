# Random forest开发集嵌套交叉验证

生成时间：2026-09-11T10:20:26Z  
队列：四分类development set，756个独立病例  
验证：5-fold outer × 5-fold inner stratified CV  
特征筛选：低表达过滤与高方差基因选择均仅在当前训练折拟合  
类别权重：`class_weight="balanced"`  
locked test：0例用于拟合、选择、预测或评分

## 主要结果

| 模型 | Macro F1 | Balanced accuracy | Accuracy | Macro OvR ROC-AUC | Log-loss | 折内选择基因 |
|---|---:|---:|---:|---:|---:|---:|
| Random forest | 0.905 ± 0.038 | 0.905 ± 0.041 | 0.917 ± 0.020 | 0.986 ± 0.005 | 0.367 ± 0.016 | 2000 |

五个外层折均选择同一组超参数：`variance_selector__k=2000`、`n_estimators=750`、`max_depth=16`、`min_samples_leaf=1`、`max_features=sqrt`。

## OOF逐类别表现

| PAM50 subtype | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Luminal A | 0.951 | 0.922 | 0.936 | 399 |
| Luminal B | 0.784 | 0.873 | 0.826 | 158 |
| Basal-like | 0.993 | 0.985 | 0.989 | 137 |
| HER2-enriched | 0.912 | 0.839 | 0.874 | 62 |

## 与线性模型的同折比较

- 相对elastic-net logistic：macro F1平均差+0.015，balanced accuracy平均差+0.004，accuracy平均差+0.019；仅2/5折macro F1更高。
- 相对LinearSVC：macro F1平均差+0.033，balanced accuracy平均差+0.015，accuracy平均差+0.037；5/5折macro F1更高。
- 相对L2 logistic：macro F1平均差+0.039，balanced accuracy平均差+0.024，accuracy平均差+0.044；4/5折macro F1更高。

Random forest目前具有最高的平均macro F1点估计，但相对elastic-net的增益较小且仅在2/5个同一外层折获胜，不能视为稳定优势。其log-loss为0.367，劣于elastic-net的0.268；概率区分度高，但概率质量并未同步改善。

## 解释与边界

- 五折均选择候选网格上限2,000个基因，提示最终冻结前应把更大特征数作为明确标记的补充敏感性分析；已观察结果后不得把扩展网格追溯称为原始主分析。
- 随机森林对约2,000个基因给出非零Gini impurity importance，远不如elastic-net平均171个非零基因稀疏。
- Impurity importance容易偏向可产生较多切分的特征，当前文件只用于诊断，不作为最终生物学证据。最终解释应在未参与拟合的数据上计算permutation importance，并报告跨折稳定性。
- 当前不能解锁189例测试集。应先完成PAM50基因排除和特征数边界敏感性分析，冻结最终模型与决策规则，再只评估locked test一次。
