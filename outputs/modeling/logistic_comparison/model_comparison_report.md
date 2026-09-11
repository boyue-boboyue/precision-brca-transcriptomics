# Dummy、L2与elastic-net开发集比较

生成时间：2026-09-10T16:49:24Z  
队列：四分类development set，756个独立病例  
验证：5-fold outer × 5-fold inner stratified CV  
locked test：0例用于拟合、预测或评分

## 外层交叉验证结果

| 模型 | Macro F1 | Balanced accuracy | Accuracy | Macro OvR ROC-AUC | 非零基因数 |
|---|---:|---:|---:|---:|---:|
| Dummy prior | 0.173 ± 0.001 | 0.250 ± 0.000 | 0.528 ± 0.003 | 0.500 ± 0.000 | 不适用 |
| Multinomial logistic L2 | 0.866 ± 0.043 | 0.881 ± 0.042 | 0.873 ± 0.041 | 0.981 ± 0.008 | 1000 |
| Multinomial logistic elastic-net | 0.890 ± 0.053 | 0.901 ± 0.050 | 0.898 ± 0.044 | 0.985 ± 0.006 | 170.8 ± 56.6 |

## 结论

- L2相对Dummy的mean macro F1提高0.693，说明表达信号远高于类别先验基线。
- Elastic-net相对L2的外层fold配对macro F1平均提高0.024，balanced accuracy平均提高0.020；5折中有4折的macro F1更高。
- Elastic-net平均仅保留170.8/1,000个非零基因，范围128–252，同时点估计性能高于L2。
- 这支持将elastic-net作为当前逻辑回归家族的首选候选模型，但最终模型仍需与预设Linear SVM和随机森林比较。
- 外层折只有5个且训练集彼此重叠，因此配对差值仅作描述，不作独立样本显著性检验。
