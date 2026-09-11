#!/usr/bin/env python3
"""Summarize RandomForest nested CV and compare completed development models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
RF_DIR = ROOT / "outputs" / "modeling" / "random_forest"
LOGISTIC_DIR = ROOT / "outputs" / "modeling" / "logistic_comparison"
SVC_DIR = ROOT / "outputs" / "modeling" / "linear_svc"
CLASS_ORDER = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rf_summary = pd.read_csv(RF_DIR / "model_summary.tsv", sep="\t").iloc[0]
    svc_summary = pd.read_csv(SVC_DIR / "model_summary.tsv", sep="\t").iloc[0]
    logistic_summary = pd.read_csv(
        LOGISTIC_DIR / "model_summary.tsv", sep="\t"
    ).set_index("model")

    comparison_records: list[dict[str, object]] = []
    for model in [
        "dummy_prior",
        "multinomial_logistic_l2",
        "multinomial_logistic_elastic_net",
    ]:
        row = logistic_summary.loc[model]
        comparison_records.append(
            {
                "model": model,
                "macro_f1_mean": row["macro_f1_mean"],
                "macro_f1_std": row["macro_f1_std"],
                "balanced_accuracy_mean": row["balanced_accuracy_mean"],
                "balanced_accuracy_std": row["balanced_accuracy_std"],
                "accuracy_mean": row["accuracy_mean"],
                "accuracy_std": row["accuracy_std"],
                "probability_macro_ovr_roc_auc_mean": row[
                    "macro_ovr_roc_auc_mean"
                ],
                "probability_macro_ovr_roc_auc_std": row[
                    "macro_ovr_roc_auc_std"
                ],
                "multiclass_log_loss_mean": row["multiclass_log_loss_mean"],
                "multiclass_log_loss_std": row["multiclass_log_loss_std"],
                "selected_genes_mean": 1000,
                "effective_nonzero_genes_mean": row["unique_nonzero_genes_mean"],
                "probability_source": "native predict_proba",
            }
        )
    comparison_records.append(
        {
            "model": "linear_svc",
            "macro_f1_mean": svc_summary["macro_f1_mean"],
            "macro_f1_std": svc_summary["macro_f1_std"],
            "balanced_accuracy_mean": svc_summary["balanced_accuracy_mean"],
            "balanced_accuracy_std": svc_summary["balanced_accuracy_std"],
            "accuracy_mean": svc_summary["accuracy_mean"],
            "accuracy_std": svc_summary["accuracy_std"],
            "probability_macro_ovr_roc_auc_mean": svc_summary[
                "calibrated_macro_ovr_roc_auc_mean"
            ],
            "probability_macro_ovr_roc_auc_std": svc_summary[
                "calibrated_macro_ovr_roc_auc_std"
            ],
            "multiclass_log_loss_mean": svc_summary[
                "calibrated_multiclass_log_loss_mean"
            ],
            "multiclass_log_loss_std": svc_summary[
                "calibrated_multiclass_log_loss_std"
            ],
            "selected_genes_mean": 1000,
            "effective_nonzero_genes_mean": svc_summary[
                "unique_nonzero_genes_mean"
            ],
            "probability_source": "training-only CalibratedClassifierCV",
        }
    )
    comparison_records.append(
        {
            "model": "random_forest",
            "macro_f1_mean": rf_summary["macro_f1_mean"],
            "macro_f1_std": rf_summary["macro_f1_std"],
            "balanced_accuracy_mean": rf_summary["balanced_accuracy_mean"],
            "balanced_accuracy_std": rf_summary["balanced_accuracy_std"],
            "accuracy_mean": rf_summary["accuracy_mean"],
            "accuracy_std": rf_summary["accuracy_std"],
            "probability_macro_ovr_roc_auc_mean": rf_summary[
                "macro_ovr_roc_auc_mean"
            ],
            "probability_macro_ovr_roc_auc_std": rf_summary[
                "macro_ovr_roc_auc_std"
            ],
            "multiclass_log_loss_mean": rf_summary[
                "multiclass_log_loss_mean"
            ],
            "multiclass_log_loss_std": rf_summary[
                "multiclass_log_loss_std"
            ],
            "selected_genes_mean": rf_summary["selected_genes_mean"],
            "effective_nonzero_genes_mean": rf_summary[
                "nonzero_impurity_importance_genes_mean"
            ],
            "probability_source": "native predict_proba",
        }
    )
    comparison = pd.DataFrame(comparison_records)
    comparison_path = RF_DIR / "comparison_all_models.tsv"
    comparison.to_csv(comparison_path, sep="\t", index=False)

    rf_metrics = pd.read_csv(RF_DIR / "outer_fold_metrics.tsv", sep="\t").set_index(
        "outer_fold"
    )
    logistic_metrics = pd.read_csv(
        LOGISTIC_DIR / "outer_fold_metrics.tsv", sep="\t"
    )
    svc_metrics = pd.read_csv(SVC_DIR / "outer_fold_metrics.tsv", sep="\t").set_index(
        "outer_fold"
    )
    comparator_metrics = {
        "multinomial_logistic_l2": logistic_metrics.loc[
            logistic_metrics["model"].eq("multinomial_logistic_l2")
        ].set_index("outer_fold"),
        "multinomial_logistic_elastic_net": logistic_metrics.loc[
            logistic_metrics["model"].eq("multinomial_logistic_elastic_net")
        ].set_index("outer_fold"),
        "linear_svc": svc_metrics,
    }
    paired_records: list[dict[str, object]] = []
    for comparator, other in comparator_metrics.items():
        for metric in ["macro_f1", "balanced_accuracy", "accuracy"]:
            differences = rf_metrics[metric] - other[metric]
            for fold, difference in differences.items():
                paired_records.append(
                    {
                        "comparison": f"random_forest_minus_{comparator}",
                        "metric": metric,
                        "outer_fold": int(fold),
                        "random_forest_value": float(rf_metrics.loc[fold, metric]),
                        "comparator_value": float(other.loc[fold, metric]),
                        "difference": float(difference),
                        "random_forest_higher": bool(difference > 0),
                    }
                )
    paired = pd.DataFrame(paired_records)
    paired_path = RF_DIR / "paired_comparison_with_linear_models.tsv"
    paired.to_csv(paired_path, sep="\t", index=False)

    predictions = pd.read_csv(RF_DIR / "outer_fold_predictions.tsv", sep="\t")
    matrix = confusion_matrix(
        predictions["observed"], predictions["predicted"], labels=CLASS_ORDER
    )
    confusion_records: list[dict[str, object]] = []
    for observed_index, observed_label in enumerate(CLASS_ORDER):
        for predicted_index, predicted_label in enumerate(CLASS_ORDER):
            confusion_records.append(
                {
                    "model": "random_forest",
                    "observed": observed_label,
                    "predicted": predicted_label,
                    "count": int(matrix[observed_index, predicted_index]),
                    "percent_within_observed": float(
                        100
                        * matrix[observed_index, predicted_index]
                        / matrix[observed_index].sum()
                    ),
                }
            )
    confusion_path = RF_DIR / "oof_confusion_matrix.tsv"
    pd.DataFrame(confusion_records).to_csv(confusion_path, sep="\t", index=False)

    paired_summary = paired.groupby(["comparison", "metric"])["difference"].agg(
        ["mean", "std"]
    )

    def delta(comparator: str, metric: str) -> float:
        comparison_name = f"random_forest_minus_{comparator}"
        return float(paired_summary.loc[(comparison_name, metric), "mean"])

    def wins(comparator: str, metric: str = "macro_f1") -> int:
        comparison_name = f"random_forest_minus_{comparator}"
        mask = paired["comparison"].eq(comparison_name) & paired["metric"].eq(
            metric
        )
        return int(paired.loc[mask, "random_forest_higher"].sum())

    parameters = pd.read_csv(RF_DIR / "best_hyperparameters.tsv", sep="\t")
    decoded = parameters["best_parameters_json"].map(json.loads)
    same_parameters = decoded.map(
        lambda item: json.dumps(item, sort_keys=True)
    ).nunique() == 1
    chosen = decoded.iloc[0]
    per_class = pd.read_csv(RF_DIR / "oof_per_class_metrics.tsv", sep="\t")
    class_rows = "\n".join(
        f"| {row.pam50_class} | {row.precision:.3f} | {row.recall:.3f} | "
        f"{row.f1:.3f} | {int(row.support)} |"
        for row in per_class.itertuples(index=False)
    )
    report = f"""# Random forest开发集嵌套交叉验证

生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  
队列：四分类development set，756个独立病例  
验证：5-fold outer × 5-fold inner stratified CV  
特征筛选：低表达过滤与高方差基因选择均仅在当前训练折拟合  
类别权重：`class_weight=\"balanced\"`  
locked test：0例用于拟合、选择、预测或评分

## 主要结果

| 模型 | Macro F1 | Balanced accuracy | Accuracy | Macro OvR ROC-AUC | Log-loss | 折内选择基因 |
|---|---:|---:|---:|---:|---:|---:|
| Random forest | {rf_summary['macro_f1_mean']:.3f} ± {rf_summary['macro_f1_std']:.3f} | {rf_summary['balanced_accuracy_mean']:.3f} ± {rf_summary['balanced_accuracy_std']:.3f} | {rf_summary['accuracy_mean']:.3f} ± {rf_summary['accuracy_std']:.3f} | {rf_summary['macro_ovr_roc_auc_mean']:.3f} ± {rf_summary['macro_ovr_roc_auc_std']:.3f} | {rf_summary['multiclass_log_loss_mean']:.3f} ± {rf_summary['multiclass_log_loss_std']:.3f} | {rf_summary['selected_genes_mean']:.0f} |

五个外层折{'均选择同一组' if same_parameters else '未选择完全相同的'}超参数：`variance_selector__k={chosen['variance_selector__k']}`、`n_estimators={chosen['classifier__n_estimators']}`、`max_depth={chosen['classifier__max_depth']}`、`min_samples_leaf={chosen['classifier__min_samples_leaf']}`、`max_features={chosen['classifier__max_features']}`。

## OOF逐类别表现

| PAM50 subtype | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{class_rows}

## 与线性模型的同折比较

- 相对elastic-net logistic：macro F1平均差{delta('multinomial_logistic_elastic_net', 'macro_f1'):+.3f}，balanced accuracy平均差{delta('multinomial_logistic_elastic_net', 'balanced_accuracy'):+.3f}，accuracy平均差{delta('multinomial_logistic_elastic_net', 'accuracy'):+.3f}；仅{wins('multinomial_logistic_elastic_net')}/5折macro F1更高。
- 相对LinearSVC：macro F1平均差{delta('linear_svc', 'macro_f1'):+.3f}，balanced accuracy平均差{delta('linear_svc', 'balanced_accuracy'):+.3f}，accuracy平均差{delta('linear_svc', 'accuracy'):+.3f}；{wins('linear_svc')}/5折macro F1更高。
- 相对L2 logistic：macro F1平均差{delta('multinomial_logistic_l2', 'macro_f1'):+.3f}，balanced accuracy平均差{delta('multinomial_logistic_l2', 'balanced_accuracy'):+.3f}，accuracy平均差{delta('multinomial_logistic_l2', 'accuracy'):+.3f}；{wins('multinomial_logistic_l2')}/5折macro F1更高。

Random forest目前具有最高的平均macro F1点估计，但相对elastic-net的增益较小且仅在2/5个同一外层折获胜，不能视为稳定优势。其log-loss为{rf_summary['multiclass_log_loss_mean']:.3f}，劣于elastic-net的{logistic_summary.loc['multinomial_logistic_elastic_net', 'multiclass_log_loss_mean']:.3f}；概率区分度高，但概率质量并未同步改善。

## 解释与边界

- 五折均选择候选网格上限2,000个基因，提示最终冻结前应把更大特征数作为明确标记的补充敏感性分析；已观察结果后不得把扩展网格追溯称为原始主分析。
- 随机森林对约2,000个基因给出非零Gini impurity importance，远不如elastic-net平均171个非零基因稀疏。
- Impurity importance容易偏向可产生较多切分的特征，当前文件只用于诊断，不作为最终生物学证据。最终解释应在未参与拟合的数据上计算permutation importance，并报告跨折稳定性。
- 当前不能解锁189例测试集。应先完成PAM50基因排除和特征数边界敏感性分析，冻结最终模型与决策规则，再只评估locked test一次。
"""
    report_path = RF_DIR / "random_forest_report.md"
    report_path.write_text(report, encoding="utf-8")

    artifacts = {
        "comparison_all_models.tsv": sha256(comparison_path),
        "paired_comparison_with_linear_models.tsv": sha256(paired_path),
        "oof_confusion_matrix.tsv": sha256(confusion_path),
        "random_forest_report.md": sha256(report_path),
        "run_manifest.json": sha256(RF_DIR / "run_manifest.json"),
    }
    manifest = {
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_script_sha256": sha256(Path(__file__).resolve()),
        "artifacts_sha256": artifacts,
    }
    with (RF_DIR / "report_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
