#!/usr/bin/env python3
"""Create audit tables and a concise report from development OOF predictions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "logistic_comparison"
CLASS_ORDER = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]
MODEL_ORDER = [
    "dummy_prior",
    "multinomial_logistic_l2",
    "multinomial_logistic_elastic_net",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    predictions = pd.read_csv(OUTPUT_DIR / "outer_fold_predictions.tsv", sep="\t")
    summary = pd.read_csv(OUTPUT_DIR / "model_summary.tsv", sep="\t").set_index("model")
    paired = pd.read_csv(
        OUTPUT_DIR / "paired_outer_fold_differences.tsv", sep="\t"
    )
    per_class_records: list[dict[str, object]] = []
    confusion_records: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        subset = predictions.loc[predictions["model"].eq(model)]
        precision, recall, f1, support = precision_recall_fscore_support(
            subset["observed"],
            subset["predicted"],
            labels=CLASS_ORDER,
            zero_division=0,
        )
        for index, subtype in enumerate(CLASS_ORDER):
            per_class_records.append(
                {
                    "model": model,
                    "pam50_class": subtype,
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
            )
        matrix = confusion_matrix(
            subset["observed"], subset["predicted"], labels=CLASS_ORDER
        )
        for true_index, true_label in enumerate(CLASS_ORDER):
            for predicted_index, predicted_label in enumerate(CLASS_ORDER):
                confusion_records.append(
                    {
                        "model": model,
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
    per_class = pd.DataFrame(per_class_records)
    confusion = pd.DataFrame(confusion_records)
    per_class_path = OUTPUT_DIR / "oof_per_class_metrics.tsv"
    confusion_path = OUTPUT_DIR / "oof_confusion_matrix.tsv"
    per_class.to_csv(per_class_path, sep="\t", index=False)
    confusion.to_csv(confusion_path, sep="\t", index=False)

    delta = paired.loc[
        paired["comparison"].eq(
            "multinomial_logistic_elastic_net_minus_multinomial_logistic_l2"
        )
    ].groupby("metric")["difference"].agg(["mean", "std"])
    l2 = summary.loc["multinomial_logistic_l2"]
    elastic = summary.loc["multinomial_logistic_elastic_net"]
    elastic_folds_better = int(
        (
            paired.loc[
                paired["comparison"].eq(
                    "multinomial_logistic_elastic_net_minus_multinomial_logistic_l2"
                )
                & paired["metric"].eq("macro_f1"),
                "difference",
            ]
            > 0
        ).sum()
    )
    elastic_counts = pd.read_csv(
        OUTPUT_DIR / "outer_fold_metrics.tsv", sep="\t"
    ).loc[
        lambda frame: frame["model"].eq("multinomial_logistic_elastic_net"),
        "unique_nonzero_genes",
    ]
    report = f"""# Dummy、L2与elastic-net开发集比较

生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  
队列：四分类development set，756个独立病例  
验证：5-fold outer × 5-fold inner stratified CV  
locked test：0例用于拟合、预测或评分

## 外层交叉验证结果

| 模型 | Macro F1 | Balanced accuracy | Accuracy | Macro OvR ROC-AUC | 非零基因数 |
|---|---:|---:|---:|---:|---:|
| Dummy prior | {summary.loc['dummy_prior', 'macro_f1_mean']:.3f} ± {summary.loc['dummy_prior', 'macro_f1_std']:.3f} | {summary.loc['dummy_prior', 'balanced_accuracy_mean']:.3f} ± {summary.loc['dummy_prior', 'balanced_accuracy_std']:.3f} | {summary.loc['dummy_prior', 'accuracy_mean']:.3f} ± {summary.loc['dummy_prior', 'accuracy_std']:.3f} | {summary.loc['dummy_prior', 'macro_ovr_roc_auc_mean']:.3f} ± {summary.loc['dummy_prior', 'macro_ovr_roc_auc_std']:.3f} | 不适用 |
| Multinomial logistic L2 | {l2['macro_f1_mean']:.3f} ± {l2['macro_f1_std']:.3f} | {l2['balanced_accuracy_mean']:.3f} ± {l2['balanced_accuracy_std']:.3f} | {l2['accuracy_mean']:.3f} ± {l2['accuracy_std']:.3f} | {l2['macro_ovr_roc_auc_mean']:.3f} ± {l2['macro_ovr_roc_auc_std']:.3f} | {l2['unique_nonzero_genes_mean']:.0f} |
| Multinomial logistic elastic-net | {elastic['macro_f1_mean']:.3f} ± {elastic['macro_f1_std']:.3f} | {elastic['balanced_accuracy_mean']:.3f} ± {elastic['balanced_accuracy_std']:.3f} | {elastic['accuracy_mean']:.3f} ± {elastic['accuracy_std']:.3f} | {elastic['macro_ovr_roc_auc_mean']:.3f} ± {elastic['macro_ovr_roc_auc_std']:.3f} | {elastic['unique_nonzero_genes_mean']:.1f} ± {elastic['unique_nonzero_genes_std']:.1f} |

## 结论

- L2相对Dummy的mean macro F1提高{l2['macro_f1_mean'] - summary.loc['dummy_prior', 'macro_f1_mean']:.3f}，说明表达信号远高于类别先验基线。
- Elastic-net相对L2的外层fold配对macro F1平均提高{delta.loc['macro_f1', 'mean']:.3f}，balanced accuracy平均提高{delta.loc['balanced_accuracy', 'mean']:.3f}；5折中有{elastic_folds_better}折的macro F1更高。
- Elastic-net平均仅保留{elastic_counts.mean():.1f}/1,000个非零基因，范围{int(elastic_counts.min())}–{int(elastic_counts.max())}，同时点估计性能高于L2。
- 这支持将elastic-net作为当前逻辑回归家族的首选候选模型，但最终模型仍需与预设Linear SVM和随机森林比较。
- 外层折只有5个且训练集彼此重叠，因此配对差值仅作描述，不作独立样本显著性检验。
"""
    report_path = OUTPUT_DIR / "model_comparison_report.md"
    report_path.write_text(report, encoding="utf-8")

    artifacts = {
        "oof_per_class_metrics.tsv": sha256(per_class_path),
        "oof_confusion_matrix.tsv": sha256(confusion_path),
        "model_comparison_report.md": sha256(report_path),
        "run_manifest.json": sha256(OUTPUT_DIR / "run_manifest.json"),
    }
    manifest = {
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_script_sha256": sha256(Path(__file__).resolve()),
        "artifacts_sha256": artifacts,
    }
    with (OUTPUT_DIR / "report_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
