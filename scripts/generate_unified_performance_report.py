#!/usr/bin/env python3
"""Generate standardized OOF and final-test tables, curves, figures and report."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "performance_report"
FIGURE_DIR = OUTPUT_DIR / "figures"
FINAL_DIR = ROOT / "outputs" / "final_evaluation"
FINAL_LOCK_PATH = ROOT / "config" / "final_model_lock_v1.json"
CLASS_ORDER = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]
MODEL_ORDER = [
    "dummy_prior",
    "multinomial_logistic_l2",
    "multinomial_logistic_elastic_net",
    "linear_svc",
    "random_forest",
]
MODEL_LABELS = {
    "dummy_prior": "Dummy prior",
    "multinomial_logistic_l2": "Logistic L2",
    "multinomial_logistic_elastic_net": "Elastic-net",
    "linear_svc": "LinearSVC",
    "random_forest": "Random forest",
}
SCHEME_LABELS = {
    "pam50_included": "PAM50 included",
    "pam50_excluded": "PAM50 excluded",
}
CLASS_COLORS = {
    "Luminal A": "#2f6fb0",
    "Luminal B": "#dc7f24",
    "Basal-like": "#3a9d5d",
    "HER2-enriched": "#b84a62",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_label(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def probability_columns() -> list[str]:
    return [f"probability_{safe_label(label)}" for label in CLASS_ORDER]


def load_all_oof_predictions() -> pd.DataFrame:
    logistic = pd.read_csv(
        ROOT / "outputs/modeling/logistic_comparison/outer_fold_predictions.tsv",
        sep="\t",
    )
    svc = pd.read_csv(
        ROOT / "outputs/modeling/linear_svc/outer_fold_predictions.tsv", sep="\t"
    )
    forest = pd.read_csv(
        ROOT / "outputs/modeling/random_forest/outer_fold_predictions.tsv", sep="\t"
    )
    included = pd.concat([logistic, svc, forest], ignore_index=True)
    included["feature_scheme"] = "pam50_included"
    excluded = pd.read_csv(
        ROOT / "outputs/modeling/pam50_excluded/outer_fold_predictions.tsv",
        sep="\t",
    )
    columns = [
        "feature_scheme",
        "model",
        "outer_fold",
        "matrix_row",
        "case_barcode",
        "sample_barcode",
        "observed",
        "predicted",
        *probability_columns(),
    ]
    combined = pd.concat([included[columns], excluded[columns]], ignore_index=True)
    combined["feature_scheme"] = pd.Categorical(
        combined["feature_scheme"], ["pam50_included", "pam50_excluded"], ordered=True
    )
    combined["model"] = pd.Categorical(combined["model"], MODEL_ORDER, ordered=True)
    return combined.sort_values(
        ["feature_scheme", "model", "outer_fold", "case_barcode"]
    ).reset_index(drop=True)


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["observed"].astype(str).to_numpy()
    predicted = frame["predicted"].astype(str).to_numpy()
    probabilities = frame[probability_columns()].to_numpy(float)
    binary = label_binarize(truth, classes=CLASS_ORDER)
    class_position = {label: index for index, label in enumerate(CLASS_ORDER)}
    true_positions = np.asarray([class_position[label] for label in truth], dtype=int)
    log_loss = float(
        -np.log(np.clip(probabilities[np.arange(len(frame)), true_positions], 1e-15, 1)).mean()
    )
    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_ovr_roc_auc": float(
            roc_auc_score(binary, probabilities, average="macro")
        ),
        "macro_average_precision": float(
            average_precision_score(binary, probabilities, average="macro")
        ),
        "multiclass_log_loss": log_loss,
    }


def make_nested_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_records = []
    for (scheme, model, fold), frame in predictions.groupby(
        ["feature_scheme", "model", "outer_fold"], observed=True, sort=False
    ):
        fold_records.append(
            {
                "feature_scheme": str(scheme),
                "model": str(model),
                "outer_fold": int(fold),
                "n": len(frame),
                **metric_values(frame),
            }
        )
    fold_metrics = pd.DataFrame(fold_records)
    metric_names = [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "macro_ovr_roc_auc",
        "macro_average_precision",
        "multiclass_log_loss",
    ]
    summary = fold_metrics.groupby(
        ["feature_scheme", "model"], sort=False
    )[metric_names].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return fold_metrics, summary.reset_index()


def make_oof_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_class_records = []
    confusion_records = []
    roc_records = []
    pr_records = []
    curve_records = []
    for (scheme, model), frame in predictions.groupby(
        ["feature_scheme", "model"], observed=True, sort=False
    ):
        truth = frame["observed"].astype(str).to_numpy()
        predicted = frame["predicted"].astype(str).to_numpy()
        probabilities = frame[probability_columns()].to_numpy(float)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=CLASS_ORDER, zero_division=0
        )
        binary = label_binarize(truth, classes=CLASS_ORDER)
        matrix = confusion_matrix(truth, predicted, labels=CLASS_ORDER)
        for observed_index, observed in enumerate(CLASS_ORDER):
            row_total = int(matrix[observed_index].sum())
            for predicted_index, predicted_class in enumerate(CLASS_ORDER):
                count = int(matrix[observed_index, predicted_index])
                confusion_records.append(
                    {
                        "feature_scheme": str(scheme),
                        "model": str(model),
                        "observed": observed,
                        "predicted": predicted_class,
                        "count": count,
                        "row_fraction": count / row_total,
                    }
                )
        for index, label in enumerate(CLASS_ORDER):
            scores = probabilities[:, index]
            auc = float(roc_auc_score(binary[:, index], scores))
            ap = float(average_precision_score(binary[:, index], scores))
            prevalence = float(binary[:, index].mean())
            per_class_records.append(
                {
                    "feature_scheme": str(scheme),
                    "model": str(model),
                    "pam50_class": label,
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
            )
            roc_records.append(
                {
                    "feature_scheme": str(scheme),
                    "model": str(model),
                    "pam50_class": label,
                    "ovr_roc_auc": auc,
                    "support": int(support[index]),
                }
            )
            pr_records.append(
                {
                    "feature_scheme": str(scheme),
                    "model": str(model),
                    "pam50_class": label,
                    "average_precision": ap,
                    "prevalence": prevalence,
                    "support": int(support[index]),
                }
            )
            fpr, tpr, _ = roc_curve(binary[:, index], scores)
            for point, (x, y) in enumerate(zip(fpr, tpr, strict=True)):
                curve_records.append(
                    {
                        "feature_scheme": str(scheme),
                        "model": str(model),
                        "curve": "roc",
                        "pam50_class": label,
                        "point": point,
                        "x": float(x),
                        "y": float(y),
                    }
                )
            pr_precision, pr_recall, _ = precision_recall_curve(binary[:, index], scores)
            for point, (x, y) in enumerate(
                zip(pr_recall, pr_precision, strict=True)
            ):
                curve_records.append(
                    {
                        "feature_scheme": str(scheme),
                        "model": str(model),
                        "curve": "precision_recall",
                        "pam50_class": label,
                        "point": point,
                        "x": float(x),
                        "y": float(y),
                    }
                )
    return (
        pd.DataFrame(per_class_records),
        pd.DataFrame(confusion_records),
        pd.DataFrame(roc_records),
        pd.DataFrame(pr_records),
        pd.DataFrame(curve_records),
    )


def paired_sensitivity(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model in MODEL_ORDER:
        subset = fold_metrics.loc[fold_metrics["model"].eq(model)]
        included = subset.loc[
            subset["feature_scheme"].eq("pam50_included")
        ].set_index("outer_fold")
        excluded = subset.loc[
            subset["feature_scheme"].eq("pam50_excluded")
        ].set_index("outer_fold")
        for metric in [
            "macro_f1",
            "balanced_accuracy",
            "macro_ovr_roc_auc",
            "macro_average_precision",
        ]:
            difference = excluded[metric] - included[metric]
            records.append(
                {
                    "model": model,
                    "metric": metric,
                    "direction": "pam50_excluded_minus_included",
                    "mean_paired_difference": float(difference.mean()),
                    "std_paired_difference": float(difference.std(ddof=1)),
                    "folds_excluded_higher": int((difference > 0).sum()),
                    "folds_equal": int((difference == 0).sum()),
                    "folds_included_higher": int((difference < 0).sum()),
                }
            )
    return pd.DataFrame(records)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
        }
    )


def plot_nested_performance(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    x = np.arange(len(MODEL_ORDER))
    offsets = {"pam50_included": -0.11, "pam50_excluded": 0.11}
    colors = {"pam50_included": "#2f6fb0", "pam50_excluded": "#dc7f24"}
    for axis, metric, title in zip(
        axes,
        ["macro_f1", "balanced_accuracy"],
        ["Nested-CV macro F1", "Nested-CV balanced accuracy"],
        strict=True,
    ):
        for scheme in ["pam50_included", "pam50_excluded"]:
            subset = summary.loc[summary["feature_scheme"].eq(scheme)].set_index("model")
            mean = subset.loc[MODEL_ORDER, f"{metric}_mean"].to_numpy()
            std = subset.loc[MODEL_ORDER, f"{metric}_std"].to_numpy()
            axis.errorbar(
                x + offsets[scheme],
                mean,
                yerr=std,
                fmt="o",
                capsize=3,
                color=colors[scheme],
                label=SCHEME_LABELS[scheme],
            )
        axis.set_title(title)
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=25, ha="right")
        axis.set_ylim(0.1, 1.01)
        axis.set_ylabel("Mean across 5 outer folds ± SD")
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_nested_cv_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(confusions: pd.DataFrame, scheme: str, filename: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.1))
    for axis, model in zip(axes.flat, MODEL_ORDER, strict=False):
        subset = confusions.loc[
            confusions["feature_scheme"].eq(scheme) & confusions["model"].eq(model)
        ]
        fractions = subset.pivot(index="observed", columns="predicted", values="row_fraction").loc[
            CLASS_ORDER, CLASS_ORDER
        ]
        counts = subset.pivot(index="observed", columns="predicted", values="count").loc[
            CLASS_ORDER, CLASS_ORDER
        ]
        image = axis.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for row in range(4):
            for column in range(4):
                value = fractions.iloc[row, column]
                color = "white" if value > 0.55 else "#222222"
                axis.text(
                    column,
                    row,
                    f"{int(counts.iloc[row, column])}\n{value:.0%}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
        axis.set_title(MODEL_LABELS[model])
        axis.set_xticks(range(4), ["LumA", "LumB", "Basal", "HER2"], rotation=30)
        axis.set_yticks(range(4), ["LumA", "LumB", "Basal", "HER2"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Observed")
        axis.grid(False)
    axes.flat[-1].axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Row fraction")
    fig.suptitle(f"Aggregated out-of-fold confusion matrices — {SCHEME_LABELS[scheme]}")
    fig.subplots_adjust(top=0.91, wspace=0.35, hspace=0.4)
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_curves(
    curves: pd.DataFrame,
    metric_table: pd.DataFrame,
    scheme: str,
    curve: str,
    filename: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.1))
    for axis, model in zip(axes.flat, MODEL_ORDER, strict=False):
        for label in CLASS_ORDER:
            points = curves.loc[
                curves["feature_scheme"].eq(scheme)
                & curves["model"].eq(model)
                & curves["curve"].eq(curve)
                & curves["pam50_class"].eq(label)
            ]
            metric_row = metric_table.loc[
                metric_table["feature_scheme"].eq(scheme)
                & metric_table["model"].eq(model)
                & metric_table["pam50_class"].eq(label)
            ].iloc[0]
            value = (
                metric_row["ovr_roc_auc"]
                if curve == "roc"
                else metric_row["average_precision"]
            )
            axis.plot(
                points["x"],
                points["y"],
                color=CLASS_COLORS[label],
                linewidth=1.5,
                label=f"{label} ({value:.3f})",
            )
            if curve == "precision_recall":
                axis.axhline(
                    metric_row["prevalence"],
                    color=CLASS_COLORS[label],
                    linewidth=0.5,
                    alpha=0.35,
                )
        if curve == "roc":
            axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=0.8)
            axis.set_xlabel("False-positive rate")
            axis.set_ylabel("True-positive rate")
            legend_title = "Class (AUC)"
        else:
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            legend_title = "Class (AP)"
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.set_title(MODEL_LABELS[model])
        axis.legend(title=legend_title, frameon=False, loc="lower left")
    axes.flat[-1].axis("off")
    curve_title = "ROC" if curve == "roc" else "Precision–recall"
    fig.suptitle(f"Aggregated out-of-fold {curve_title} curves — {SCHEME_LABELS[scheme]}")
    fig.subplots_adjust(top=0.91, wspace=0.35, hspace=0.4)
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_delta(differences: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    x = np.arange(len(MODEL_ORDER))
    for axis, metric, title in zip(
        axes,
        ["macro_f1", "balanced_accuracy"],
        ["Macro F1 change", "Balanced accuracy change"],
        strict=True,
    ):
        subset = differences.loc[differences["metric"].eq(metric)].set_index("model")
        values = subset.loc[MODEL_ORDER, "mean_paired_difference"].to_numpy()
        errors = subset.loc[MODEL_ORDER, "std_paired_difference"].to_numpy()
        axis.bar(x, values, color=np.where(values >= 0, "#3a9d5d", "#b84a62"), alpha=0.85)
        axis.errorbar(x, values, yerr=errors, fmt="none", color="#333333", capsize=3)
        axis.axhline(0, color="#333333", linewidth=0.9)
        axis.set_title(title)
        axis.set_ylabel("Excluded − included (paired folds), mean ± SD")
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_pam50_exclusion_delta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_test() -> None:
    confusion = pd.read_csv(FINAL_DIR / "locked_test_confusion_matrix.tsv", sep="\t").set_index("observed").loc[
        CLASS_ORDER, CLASS_ORDER
    ]
    per_class = pd.read_csv(FINAL_DIR / "locked_test_per_class_metrics.tsv", sep="\t").set_index("pam50_class")
    curves = pd.read_csv(FINAL_DIR / "locked_test_curve_points.tsv", sep="\t")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    fractions = confusion.div(confusion.sum(axis=1), axis=0)
    image = axes[0].imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    for row in range(4):
        for column in range(4):
            value = fractions.iloc[row, column]
            axes[0].text(
                column,
                row,
                f"{int(confusion.iloc[row, column])}\n{value:.0%}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "#222222",
                fontsize=7,
            )
    axes[0].set_xticks(range(4), ["LumA", "LumB", "Basal", "HER2"], rotation=30)
    axes[0].set_yticks(range(4), ["LumA", "LumB", "Basal", "HER2"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Observed")
    axes[0].set_title("Locked test confusion matrix")
    axes[0].grid(False)
    fig.colorbar(image, ax=axes[0], fraction=0.047, pad=0.04)
    for label in CLASS_ORDER:
        roc_points = curves.loc[
            curves["curve"].eq("roc") & curves["pam50_class"].eq(label)
        ]
        pr_points = curves.loc[
            curves["curve"].eq("precision_recall") & curves["pam50_class"].eq(label)
        ]
        axes[1].plot(
            roc_points["x"], roc_points["y"], color=CLASS_COLORS[label],
            label=f"{label} ({per_class.loc[label, 'ovr_roc_auc']:.3f})"
        )
        axes[2].plot(
            pr_points["x"], pr_points["y"], color=CLASS_COLORS[label],
            label=f"{label} ({per_class.loc[label, 'average_precision']:.3f})"
        )
        axes[2].axhline(
            per_class.loc[label, "prevalence"], color=CLASS_COLORS[label], alpha=0.3, linewidth=0.5
        )
    axes[1].plot([0, 1], [0, 1], "--", color="#777777", linewidth=0.8)
    axes[1].set(title="Locked test OvR ROC", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
    axes[2].set(title="Locked test precision–recall", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend(title="Class (AUC)", frameon=False, loc="lower right", fontsize=7)
    axes[2].legend(title="Class (AP)", frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_locked_test_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=["number"]).columns:
        if pd.api.types.is_integer_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )
        else:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
            )
    headers = [str(column) for column in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    roc_auc: pd.DataFrame,
    pr_auc: pd.DataFrame,
    differences: pd.DataFrame,
) -> None:
    final_summary = pd.read_csv(FINAL_DIR / "locked_test_summary.tsv", sep="\t")
    final_per_class = pd.read_csv(FINAL_DIR / "locked_test_per_class_metrics.tsv", sep="\t")
    bootstrap_ci = pd.read_csv(FINAL_DIR / "bootstrap_confidence_intervals.tsv", sep="\t")
    lock = json.loads(FINAL_LOCK_PATH.read_text(encoding="utf-8"))

    primary = summary.loc[summary["feature_scheme"].eq("pam50_included")].copy()
    primary["model"] = primary["model"].map(MODEL_LABELS)
    primary_display = primary[
        [
            "model",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "macro_ovr_roc_auc_mean",
            "macro_average_precision_mean",
        ]
    ].rename(
        columns={
            "model": "模型",
            "macro_f1_mean": "macro F1 均值",
            "macro_f1_std": "macro F1 SD",
            "balanced_accuracy_mean": "balanced accuracy 均值",
            "balanced_accuracy_std": "balanced accuracy SD",
            "macro_ovr_roc_auc_mean": "macro OvR AUC",
            "macro_average_precision_mean": "macro PR-AUC/AP",
        }
    )
    selected = lock["selected_model"]
    margin = lock["selection_margin_over_runner_up_macro_f1"]
    selected_ci = bootstrap_ci.loc[
        bootstrap_ci["pam50_class"].eq("macro_or_overall")
        & bootstrap_ci["metric"].isin(["macro_f1", "balanced_accuracy"])
    ][["metric", "estimate", "ci_lower_95", "ci_upper_95"]]
    smaller = final_per_class.sort_values("support").head(2)[
        ["pam50_class", "support", "recall", "f1", "average_precision"]
    ]
    included_detail = (
        per_class.loc[per_class["feature_scheme"].eq("pam50_included")]
        .merge(
            roc_auc.loc[roc_auc["feature_scheme"].eq("pam50_included")][
                ["model", "pam50_class", "ovr_roc_auc"]
            ],
            on=["model", "pam50_class"],
        )
        .merge(
            pr_auc.loc[pr_auc["feature_scheme"].eq("pam50_included")][
                ["model", "pam50_class", "average_precision", "prevalence"]
            ],
            on=["model", "pam50_class"],
        )
    )
    included_detail["model"] = included_detail["model"].map(MODEL_LABELS)
    included_detail = included_detail[
        [
            "model", "pam50_class", "support", "precision", "recall", "f1",
            "ovr_roc_auc", "average_precision", "prevalence",
        ]
    ].rename(
        columns={
            "model": "模型", "pam50_class": "类别", "support": "n",
            "f1": "F1", "ovr_roc_auc": "OvR ROC-AUC",
            "average_precision": "PR-AUC/AP", "prevalence": "类别比例",
        }
    )
    delta_primary = differences.loc[
        differences["metric"].isin(["macro_f1", "balanced_accuracy"])
    ].copy()
    delta_primary["model"] = delta_primary["model"].map(MODEL_LABELS)
    delta_primary = delta_primary[
        ["model", "metric", "mean_paired_difference", "std_paired_difference", "folds_excluded_higher"]
    ].rename(
        columns={
            "model": "模型",
            "metric": "指标",
            "mean_paired_difference": "excluded − included 均值",
            "std_paired_difference": "配对差 SD",
            "folds_excluded_higher": "excluded 更高折数/5",
        }
    )
    final_summary_display = final_summary[
        ["balanced_accuracy", "macro_f1", "accuracy", "macro_ovr_roc_auc", "macro_average_precision", "multiclass_log_loss"]
    ].T.reset_index()
    final_summary_display.columns = ["指标", "locked test 估计"]

    report = f"""# OncoStratify-BRCA 统一性能报告

**生成时间：** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**主分析：** 四分类 PAM50，756例 development，189例 locked test  
**预先指定的模型选择指标：** outer nested-CV mean macro F1；balanced accuracy 为共同主要报告指标。

## 1. 结论

在 PAM50-included 主分析中，按预注册规则选择 **{MODEL_LABELS[selected]}**。其 outer-fold mean macro F1 比第二名高 **{margin:.3f}**，超过0.01平局阈值，因此未启用简约性平局规则。所有候选模型的比较仅使用 development OOF 结果；最终模型冻结后，locked test 只评估一次。

locked test 的两个主要指标及病例级分层 bootstrap 95% percentile CI：

{markdown_table(selected_ci.rename(columns={'metric':'指标','estimate':'估计','ci_lower_95':'95% CI下限','ci_upper_95':'95% CI上限'}))}

## 2. PAM50-included nested-CV

下表均为5个 outer folds 的均值和样本标准差。分类标签来自原始分类器；LinearSVC 的 ROC/PR 概率来自每个 outer-training partition 内完成的 sigmoid 校准。

{markdown_table(primary_display)}

![Nested-CV performance](figures/01_nested_cv_performance.png)

## 3. 聚合 OOF 逐类性能

完整逐类 precision、recall、F1、support 见 `oof_per_class_metrics.tsv`；逐类 OvR ROC-AUC 与 PR-AUC/AP 分别见 `oof_roc_auc.tsv` 和 `oof_pr_auc.tsv`。

{markdown_table(included_detail)}

![Included confusion matrices](figures/02_oof_confusion_included.png)

![Included ROC curves](figures/03_oof_roc_included.png)

![Included precision-recall curves](figures/04_oof_pr_included.png)

## 4. PAM50-included 与 excluded

excluded 方案在进入 Pipeline 前移除锁定的50个 PAM50 signature 基因，并保持同一病例、同一 outer/inner folds、同一模型网格。差值定义为 excluded − included；该敏感性分析不覆盖主模型排名。

{markdown_table(delta_primary)}

![PAM50 exclusion delta](figures/06_pam50_exclusion_delta.png)

excluded 的完整混淆矩阵与逐类曲线：

![Excluded confusion matrices](figures/02b_oof_confusion_excluded.png)

![Excluded ROC curves](figures/03b_oof_roc_excluded.png)

![Excluded precision-recall curves](figures/04b_oof_pr_excluded.png)

## 5. 最终 locked test

{markdown_table(final_summary_display)}

较小类别（按test support排序）的召回率、F1和 average precision：

{markdown_table(smaller.rename(columns={'pam50_class':'类别','support':'n','recall':'recall','f1':'F1','average_precision':'PR-AUC/AP'}))}

全部 locked-test 逐类结果：

{markdown_table(final_per_class.rename(columns={'pam50_class':'类别','support':'n','f1':'F1','ovr_roc_auc':'OvR ROC-AUC','average_precision':'PR-AUC/AP','prevalence':'类别比例'}))}

![Locked test performance](figures/07_locked_test_performance.png)

完整逐类指标见 `../final_evaluation/locked_test_per_class_metrics.tsv`；所有总体及逐类 bootstrap CI 见 `../final_evaluation/bootstrap_confidence_intervals.tsv`。

## 6. 解释边界

- Nested-CV 的“±”是5个 outer folds 的标准差，不是95%置信区间。
- 95% bootstrap CI 仅针对一次性 locked-test 评估，默认1,000次、以病例为单位并按真实 PAM50 类别分层。
- PR-AUC 在表中以 average precision 实现；它对 HER2-enriched 等较小类别比 accuracy 更有信息。
- 不为每个候选模型分别查看 locked test：这样会把测试集转化为额外的模型选择集，违反预先锁定的一次性评估政策。
- PAM50 标签由表达模式定义，因此高性能表示对既有 PAM50 分型的复现，不等同于独立临床获益预测。

## 7. 可复现性

原始逐病例 OOF 概率、逐折指标、curve points、混淆矩阵、PAM50排除差异和最终测试结果均保存为 TSV；`run_manifest.json`记录输入与输出哈希。PAM50 signature 来源固定到 `genefu` commit `{json.loads((ROOT / 'config/pam50_exclusion_v1.json').read_text())['source']['commit']}`。
"""
    (OUTPUT_DIR / "unified_performance_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    predictions = load_all_oof_predictions()
    fold_metrics, summary = make_nested_tables(predictions)
    per_class, confusions, roc_auc, pr_auc, curves = make_oof_tables(predictions)
    differences = paired_sensitivity(fold_metrics)
    fold_metrics.to_csv(OUTPUT_DIR / "nested_cv_fold_metrics.tsv", sep="\t", index=False)
    summary.to_csv(OUTPUT_DIR / "nested_cv_summary.tsv", sep="\t", index=False)
    per_class.to_csv(OUTPUT_DIR / "oof_per_class_metrics.tsv", sep="\t", index=False)
    confusions.to_csv(OUTPUT_DIR / "oof_confusion_matrices.tsv", sep="\t", index=False)
    roc_auc.to_csv(OUTPUT_DIR / "oof_roc_auc.tsv", sep="\t", index=False)
    pr_auc.to_csv(OUTPUT_DIR / "oof_pr_auc.tsv", sep="\t", index=False)
    curves.to_csv(
        OUTPUT_DIR / "oof_curve_points.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    differences.to_csv(
        OUTPUT_DIR / "pam50_included_excluded_differences.tsv", sep="\t", index=False
    )
    final_lock = json.loads(FINAL_LOCK_PATH.read_text(encoding="utf-8"))
    selection = pd.DataFrame(final_lock["candidate_ranking"])
    selection.insert(0, "rank", np.arange(1, len(selection) + 1))
    selection["selected"] = selection["model"].eq(final_lock["selected_model"])
    selection["selection_metric"] = "mean_outer_fold_macro_f1"
    selection["tie_threshold_macro_f1"] = final_lock["selection_rule"][
        "tie_threshold_macro_f1"
    ]
    selection.to_csv(OUTPUT_DIR / "model_selection.tsv", sep="\t", index=False)
    plot_nested_performance(summary)
    plot_confusions(confusions, "pam50_included", "02_oof_confusion_included.png")
    plot_confusions(confusions, "pam50_excluded", "02b_oof_confusion_excluded.png")
    plot_curves(curves, roc_auc, "pam50_included", "roc", "03_oof_roc_included.png")
    plot_curves(curves, roc_auc, "pam50_excluded", "roc", "03b_oof_roc_excluded.png")
    plot_curves(curves, pr_auc, "pam50_included", "precision_recall", "04_oof_pr_included.png")
    plot_curves(curves, pr_auc, "pam50_excluded", "precision_recall", "04b_oof_pr_excluded.png")
    plot_sensitivity_delta(differences)
    plot_final_test()
    write_report(summary, per_class, roc_auc, pr_auc, differences)

    artifact_paths = [
        path
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file()
        and path.name not in {"run_manifest.json", "verification_report.json"}
    ]
    inputs = [
        ROOT / "outputs/modeling/logistic_comparison/outer_fold_predictions.tsv",
        ROOT / "outputs/modeling/linear_svc/outer_fold_predictions.tsv",
        ROOT / "outputs/modeling/random_forest/outer_fold_predictions.tsv",
        ROOT / "outputs/modeling/pam50_excluded/outer_fold_predictions.tsv",
        FINAL_DIR / "run_manifest.json",
        FINAL_LOCK_PATH,
    ]
    manifest = {
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": MODEL_ORDER,
        "feature_schemes": ["pam50_included", "pam50_excluded"],
        "development_n_per_model_scheme": 756,
        "selected_final_model": "random_forest",
        "locked_test_n": 189,
        "locked_test_evaluation_count": 1,
        "inputs_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs_sha256": {
            str(path.relative_to(OUTPUT_DIR)): sha256(path) for path in artifact_paths
        },
        "generator_sha256": sha256(Path(__file__).resolve()),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
