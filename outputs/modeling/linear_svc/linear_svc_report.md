# LinearSVC开发集嵌套交叉验证

生成时间：2026-09-10T17:05:47Z  
队列：四分类development set，756个独立病例  
验证：5-fold outer × 5-fold inner stratified CV  
分类器：LinearSVC；RBF模型拟合数为0  
概率：每个outer-training set内部的5-fold CalibratedClassifierCV sigmoid校准  
locked test：0例用于拟合、校准、预测或评分

## 结果

| 模型 | Macro F1 | Balanced accuracy | Accuracy | 概率ROC-AUC | Log-loss | 非零基因 |
|---|---:|---:|---:|---:|---:|---:|
| LinearSVC | 0.872 ± 0.034 | 0.890 ± 0.042 | 0.880 ± 0.024 | 0.977 ± 0.010 | 0.321 ± 0.051 | 1000 |

## 与逻辑回归比较

- LinearSVC相对L2 logistic：macro F1平均差+0.006，balanced accuracy平均差+0.009。
- LinearSVC相对elastic-net logistic：macro F1平均差-0.018，balanced accuracy平均差-0.011；5折中2折macro F1更高。
- LinearSVC的L2 penalty使1,000个候选基因均具有非零权重；elastic-net logistic平均仅保留171个非零基因，因此后者更稀疏、也更便于解释。
- 最佳参数分布：C=0.001: 4折, C=0.01: 1折。较小C在多数折胜出，说明强正则化更适合当前高维表达数据。
- LinearSVC没有收敛警告。训练数据内校准提供了可用概率，但其概率ROC-AUC与log-loss不优于elastic-net logistic。

当前结果不支持引入RBF kernel。Elastic-net logistic仍是已完成候选模型中的领先者；最终选择还需等待随机森林以及预设敏感性分析。
