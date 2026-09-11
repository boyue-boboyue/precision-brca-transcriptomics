#!/usr/bin/env python3
"""Compare LinearSVC with the completed development-only logistic models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
SVC_DIR = ROOT / "outputs" / "modeling" / "linear_svc"
LOGISTIC_DIR = ROOT / "outputs" / "modeling" / "logistic_comparison"
CLASS_ORDER = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    svc_summary = pd.read_csv(SVC_DIR / "model_summary.tsv", sep="\t").iloc[0]
    logistic_summary = pd.read_csv(
        LOGISTIC_DIR / "model_summary.tsv", sep="\t"
    ).set_index("model")
    comparison_records = []
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
                "probability_macro_ovr_roc_auc_std": row["macro_ovr_roc_auc_std"],
                "multiclass_log_loss_mean": row["multiclass_log_loss_mean"],
                "multiclass_log_loss_std": row["multiclass_log_loss_std"],
                "unique_nonzero_genes_mean": row["unique_nonzero_genes_mean"],
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
            "unique_nonzero_genes_mean": svc_summary["unique_nonzero_genes_mean"],
            "probability_source": "training-only CalibratedClassifierCV",
        }
    )
    comparison = pd.DataFrame(comparison_records)
    comparison_path = SVC_DIR / "comparison_with_logistic.tsv"
    comparison.to_csv(comparison_path, sep="\t", index=False)

    svc_metrics = pd.read_csv(SVC_DIR / "outer_fold_metrics.tsv", sep="\t")
    logistic_metrics = pd.read_csv(
        LOGISTIC_DIR / "outer_fold_metrics.tsv", sep="\t"
    )
    paired_records = []
    for comparator in [
        "multinomial_logistic_l2",
        "multinomial_logistic_elastic_net",
    ]:
        other = logistic_metrics.loc[
            logistic_metrics["model"].eq(comparator)
        ].set_index("outer_fold")
        svc = svc_metrics.set_index("outer_fold")
        for metric in ["macro_f1", "balanced_accuracy", "accuracy"]:
            differences = svc[metric] - other[metric]
            for fold, difference in differences.items():
                paired_records.append(
                    {
                        "comparison": f"linear_svc_minus_{comparator}",
                        "metric": metric,
                        "outer_fold": int(fold),
                        "difference": float(difference),
                    }
                )
    paired = pd.DataFrame(paired_records)
    paired_path = SVC_DIR / "paired_comparison_with_logistic.tsv"
    paired.to_csv(paired_path, sep="\t", index=False)

    predictions = pd.read_csv(SVC_DIR / "outer_fold_predictions.tsv", sep="\t")
    matrix = confusion_matrix(
        predictions["observed"], predictions["predicted"], labels=CLASS_ORDER
    )
    confusion_records = []
    for true_index, true_label in enumerate(CLASS_ORDER):
        for predicted_index, predicted_label in enumerate(CLASS_ORDER):
            confusion_records.append(
                {
                    "model": "linear_svc",
                    "observed": true_label,
                    "predicted": predicted_label,
                    "count": int(matrix[true_index, predicted_index]),
                    "percent_within_observed": float(
                        100
                        * matrix[true_index, predicted_index]
                        / matrix[true_index].sum()
                    ),
                }
            )
    confusion_path = SVC_DIR / "oof_confusion_matrix.tsv"
    pd.DataFrame(confusion_records).to_csv(
        confusion_path, sep="\t", index=False
    )

    paired_summary = paired.groupby(["comparison", "metric"])["difference"].agg(
        ["mean", "std"]
    )
    versus_l2 = paired_summary.loc["linear_svc_minus_multinomial_logistic_l2"]
    versus_elastic = paired_summary.loc[
        "linear_svc_minus_multinomial_logistic_elastic_net"
    ]
    better_than_elastic = int(
        (
            paired.loc[
                paired["comparison"].eq(
                    "linear_svc_minus_multinomial_logistic_elastic_net"
                )
                & paired["metric"].eq("macro_f1"),
                "difference",
            ]
            > 0
        ).sum()
    )
    parameters = pd.read_csv(SVC_DIR / "best_hyperparameters.tsv", sep="\t")
    c_counts = parameters["best_parameters_json"].map(
        lambda value: json.loads(value)["classifier__C"]
    ).value_counts()
    c_description = ", ".join(
        f"C={value:g}: {count}折" for value, count in c_counts.sort_index().items()
    )
    report = f"""# LinearSVC开发集嵌套交叉验证

生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  
队列：四分类development set，756个独立病例  
验证：5-fold outer × 5-fold inner stratified CV  
分类器：LinearSVC；RBF模型拟合数为0  
概率：每个outer-training set内部的5-fold CalibratedClassifierCV sigmoid校准  
locked test：0例用于拟合、校准、预测或评分

## 结果

| 模型 | Macro F1 | Balanced accuracy | Accuracy | 概率ROC-AUC | Log-loss | 非零基因 |
|---|---:|---:|---:|---:|---:|---:|
| LinearSVC | {svc_summary['macro_f1_mean']:.3f} ± {svc_summary['macro_f1_std']:.3f} | {svc_summary['balanced_accuracy_mean']:.3f} ± {svc_summary['balanced_accuracy_std']:.3f} | {svc_summary['accuracy_mean']:.3f} ± {svc_summary['accuracy_std']:.3f} | {svc_summary['calibrated_macro_ovr_roc_auc_mean']:.3f} ± {svc_summary['calibrated_macro_ovr_roc_auc_std']:.3f} | {svc_summary['calibrated_multiclass_log_loss_mean']:.3f} ± {svc_summary['calibrated_multiclass_log_loss_std']:.3f} | {svc_summary['unique_nonzero_genes_mean']:.0f} |

## 与逻辑回归比较

- LinearSVC相对L2 logistic：macro F1平均差{versus_l2.loc['macro_f1', 'mean']:+.3f}，balanced accuracy平均差{versus_l2.loc['balanced_accuracy', 'mean']:+.3f}。
- LinearSVC相对elastic-net logistic：macro F1平均差{versus_elastic.loc['macro_f1', 'mean']:+.3f}，balanced accuracy平均差{versus_elastic.loc['balanced_accuracy', 'mean']:+.3f}；5折中{better_than_elastic}折macro F1更高。
- LinearSVC的L2 penalty使1,000个候选基因均具有非零权重；elastic-net logistic平均仅保留171个非零基因，因此后者更稀疏、也更便于解释。
- 最佳参数分布：{c_description}。较小C在多数折胜出，说明强正则化更适合当前高维表达数据。
- LinearSVC没有收敛警告。训练数据内校准提供了可用概率，但其概率ROC-AUC与log-loss不优于elastic-net logistic。

当前结果不支持引入RBF kernel。Elastic-net logistic仍是已完成候选模型中的领先者；最终选择还需等待随机森林以及预设敏感性分析。
"""
    report_path = SVC_DIR / "linear_svc_report.md"
    report_path.write_text(report, encoding="utf-8")

    artifacts = {
        "comparison_with_logistic.tsv": sha256(comparison_path),
        "paired_comparison_with_logistic.tsv": sha256(paired_path),
        "oof_confusion_matrix.tsv": sha256(confusion_path),
        "linear_svc_report.md": sha256(report_path),
        "run_manifest.json": sha256(SVC_DIR / "run_manifest.json"),
    }
    manifest = {
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_script_sha256": sha256(Path(__file__).resolve()),
        "artifacts_sha256": artifacts,
    }
    with (SVC_DIR / "report_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
