#!/usr/bin/env python3
"""Validate stable model-associated genes against expression and clinical biology."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "interpretability_v1.json"
ASSIGNMENT_PATH = (
    ROOT / "data" / "processed" / "splits" / "four_class_split_assignments.tsv"
)
GENE_PATH = ROOT / "data" / "processed" / "expression" / "genes.tsv"
MATRIX_PATH = ROOT / "data" / "processed" / "expression" / "log2_tpm_float32.npy"
ANALYSIS_SAMPLE_PATH = ROOT / "outputs" / "eda" / "tables" / "analysis_samples.tsv"
PAM50_PATH = ROOT / "data" / "processed" / "labels" / "pam50_signature_genes_v1.tsv"
INTERPRET_DIR = ROOT / "outputs" / "interpretability"
BIO_DIR = INTERPRET_DIR / "biological_validation"
CLASS_ORDER = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]
RECEPTORS = {"ER": "er_status", "PR": "pr_status", "HER2": "her2_status"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    result = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return result
    finite_p = p[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[finite] = restored
    return result


def verify_inputs(config: dict[str, Any]) -> None:
    if config["biological_validation"]["stable_gene_minimum_outer_folds"] != 3:
        raise RuntimeError("Unexpected stable-gene threshold")
    for relative, expected in config["inputs_sha256"].items():
        if relative.startswith("scripts/") or relative.startswith("requirements-"):
            if sha256(ROOT / relative) != expected:
                raise RuntimeError(f"Locked script/environment hash mismatch: {relative}")
    run_manifest = json.loads(
        (INTERPRET_DIR / "run_manifest.json").read_text(encoding="utf-8")
    )
    if run_manifest.get("status") != "COMPLETE":
        raise RuntimeError("Interpretability run is incomplete")


def stable_gene_table(config: dict[str, Any]) -> pd.DataFrame:
    minimum = int(config["biological_validation"]["stable_gene_minimum_outer_folds"])
    elastic = pd.read_csv(INTERPRET_DIR / "elastic_net_top20_stability.tsv", sep="\t")
    elastic = elastic.loc[elastic["top20_outer_fold_count"].ge(minimum)].copy()
    elastic_summary = elastic.groupby(["matrix_column", "gene_name"], as_index=False).agg(
        elastic_max_top20_fold_count=("top20_outer_fold_count", "max"),
        elastic_mean_top20_frequency=("top20_outer_fold_frequency", "mean"),
        elastic_classes=(
            "pam50_class",
            lambda values: ";".join(sorted(set(map(str, values)))),
        ),
        elastic_directions=(
            "dominant_direction",
            lambda values: ";".join(sorted(set(map(str, values)))),
        ),
    )
    forest = pd.read_csv(INTERPRET_DIR / "rf_top20_stability.tsv", sep="\t")
    forest = forest.loc[forest["top20_outer_fold_count"].ge(minimum)].rename(
        columns={
            "top20_outer_fold_count": "rf_top20_outer_fold_count",
            "top20_outer_fold_frequency": "rf_top20_outer_fold_frequency",
        }
    )
    stable = elastic_summary.merge(
        forest[
            [
                "matrix_column",
                "gene_name",
                "rf_top20_outer_fold_count",
                "rf_top20_outer_fold_frequency",
            ]
        ],
        on=["matrix_column", "gene_name"],
        how="outer",
    )
    count_columns = ["elastic_max_top20_fold_count", "rf_top20_outer_fold_count"]
    stable[count_columns] = stable[count_columns].fillna(0).astype(int)
    stable[["elastic_mean_top20_frequency", "rf_top20_outer_fold_frequency"]] = stable[
        ["elastic_mean_top20_frequency", "rf_top20_outer_fold_frequency"]
    ].fillna(0.0)
    stable["stable_source"] = np.select(
        [
            stable["elastic_max_top20_fold_count"].ge(minimum)
            & stable["rf_top20_outer_fold_count"].ge(minimum),
            stable["elastic_max_top20_fold_count"].ge(minimum),
            stable["rf_top20_outer_fold_count"].ge(minimum),
        ],
        ["Elastic-net_and_RF", "Elastic-net", "RF_permutation"],
        default="unexpected",
    )
    stable["maximum_top20_fold_count"] = stable[count_columns].max(axis=1)
    stable = stable.sort_values(
        ["maximum_top20_fold_count", "stable_source", "gene_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    stable.insert(0, "stable_gene_order", np.arange(1, len(stable) + 1))
    return stable


def expression_validation(
    stable: pd.DataFrame,
    assignments: pd.DataFrame,
    samples: pd.DataFrame,
    matrix: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    development = assignments.loc[
        assignments["holdout_split"].eq("development")
    ].reset_index(drop=True)
    metadata = development[
        ["matrix_row", "case_barcode", "sample_barcode", "pam50_standard"]
    ].merge(
        samples[
            ["matrix_row", "case_barcode", "er_status", "pr_status", "her2_status"]
        ],
        on=["matrix_row", "case_barcode"],
        how="left",
        validate="one_to_one",
    )
    columns = stable["matrix_column"].astype(int).to_numpy()
    values = np.asarray(
        matrix[metadata["matrix_row"].astype(int).to_numpy()][:, columns],
        dtype=np.float32,
    )
    long = pd.DataFrame(values, columns=stable["gene_name"].astype(str)).assign(
        matrix_row=metadata["matrix_row"].to_numpy(),
        case_barcode=metadata["case_barcode"].to_numpy(),
        sample_barcode=metadata["sample_barcode"].to_numpy(),
        pam50_class=metadata["pam50_standard"].to_numpy(),
        er_status=metadata["er_status"].to_numpy(),
        pr_status=metadata["pr_status"].to_numpy(),
        her2_status=metadata["her2_status"].to_numpy(),
    ).melt(
        id_vars=[
            "matrix_row",
            "case_barcode",
            "sample_barcode",
            "pam50_class",
            "er_status",
            "pr_status",
            "her2_status",
        ],
        var_name="gene_name",
        value_name="log2_tpm",
    )
    stable_columns = stable[["gene_name", "matrix_column"]].drop_duplicates("gene_name")
    long = long.merge(stable_columns, on="gene_name", how="left", validate="many_to_one")

    distribution = long.groupby(["gene_name", "matrix_column", "pam50_class"], as_index=False).agg(
        n=("log2_tpm", "size"),
        mean_log2_tpm=("log2_tpm", "mean"),
        sd_log2_tpm=("log2_tpm", "std"),
        median_log2_tpm=("log2_tpm", "median"),
        q25_log2_tpm=("log2_tpm", lambda values: float(np.quantile(values, 0.25))),
        q75_log2_tpm=("log2_tpm", lambda values: float(np.quantile(values, 0.75))),
    )
    subtype_tests: list[dict[str, Any]] = []
    for gene, group in long.groupby("gene_name"):
        arrays = [
            group.loc[group["pam50_class"].eq(subtype), "log2_tpm"].to_numpy()
            for subtype in CLASS_ORDER
        ]
        statistic, p_value = kruskal(*arrays)
        total_n = sum(len(values) for values in arrays)
        eta_squared = max(0.0, float((statistic - len(arrays) + 1) / (total_n - len(arrays))))
        means = {subtype: float(values.mean()) for subtype, values in zip(CLASS_ORDER, arrays)}
        subtype_tests.append(
            {
                "gene_name": gene,
                "matrix_column": int(group["matrix_column"].iloc[0]),
                "kruskal_h": float(statistic),
                "p_value": float(p_value),
                "epsilon_squared": min(1.0, eta_squared),
                "highest_mean_subtype": max(means, key=means.get),
                "lowest_mean_subtype": min(means, key=means.get),
                "mean_expression_range": max(means.values()) - min(means.values()),
            }
        )
    subtype = pd.DataFrame(subtype_tests)
    subtype["bh_q_value"] = bh_adjust(subtype["p_value"])
    subtype = subtype.sort_values(["bh_q_value", "epsilon_squared", "gene_name"], ascending=[True, False, True])
    return long, distribution, subtype


def receptor_validation(long: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for receptor, column in RECEPTORS.items():
        for gene, group in long.groupby("gene_name"):
            positive = group.loc[group[column].eq("Positive"), "log2_tpm"].to_numpy()
            negative = group.loc[group[column].eq("Negative"), "log2_tpm"].to_numpy()
            if len(positive) < 5 or len(negative) < 5:
                statistic = p_value = effect = np.nan
            else:
                statistic, p_value = mannwhitneyu(
                    positive, negative, alternative="two-sided", method="asymptotic"
                )
                effect = 2 * float(statistic) / (len(positive) * len(negative)) - 1
            records.append(
                {
                    "receptor": receptor,
                    "status_comparison": "Positive_minus_Negative",
                    "gene_name": gene,
                    "matrix_column": int(group["matrix_column"].iloc[0]),
                    "positive_n": len(positive),
                    "negative_n": len(negative),
                    "positive_mean_log2_tpm": float(positive.mean()) if len(positive) else np.nan,
                    "negative_mean_log2_tpm": float(negative.mean()) if len(negative) else np.nan,
                    "positive_median_log2_tpm": float(np.median(positive)) if len(positive) else np.nan,
                    "negative_median_log2_tpm": float(np.median(negative)) if len(negative) else np.nan,
                    "mean_difference": float(positive.mean() - negative.mean())
                    if len(positive) and len(negative)
                    else np.nan,
                    "mann_whitney_u": float(statistic) if np.isfinite(statistic) else np.nan,
                    "rank_biserial_correlation": effect,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )
    result = pd.DataFrame(records)
    result["bh_q_value"] = result.groupby("receptor", group_keys=False)["p_value"].transform(
        lambda values: bh_adjust(values)
    )
    return result.sort_values(["receptor", "bh_q_value", "gene_name"])


def classify_genes(
    stable: pd.DataFrame,
    subtype: pd.DataFrame,
    receptor: pd.DataFrame,
    pam50: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    known_markers = set(config["expected_signal_audit"])
    pam50_genes = set(pam50["current_symbol"].astype(str))
    receptor_support = receptor.loc[
        receptor["bh_q_value"].lt(0.05)
        & receptor["rank_biserial_correlation"].abs().ge(0.33)
    ]
    receptor_map = receptor_support.groupby("gene_name")["receptor"].agg(
        lambda values: ";".join(sorted(set(values)))
    )
    subtype_map = subtype.set_index("gene_name")
    records: list[dict[str, Any]] = []
    for row in stable.itertuples():
        gene = str(row.gene_name)
        is_expected = gene in known_markers
        is_pam50 = gene in pam50_genes
        supporting_receptors = receptor_map.get(gene, "")
        subtype_q = float(subtype_map.loc[gene, "bh_q_value"])
        if is_expected or is_pam50:
            category = "known_marker_or_PAM50"
            reason = "predeclared_expected_marker" if is_expected else "locked_PAM50_member"
            if is_expected and is_pam50:
                reason = "predeclared_expected_marker_and_locked_PAM50_member"
        elif supporting_receptors:
            category = "clinically_correlated_non_PAM50"
            reason = f"IHC-associated_with_{supporting_receptors}"
        elif subtype_q < 0.05:
            category = "potential_candidate_not_novelty_claim"
            reason = "stable_and_subtype_differential_but_requires_external_validation"
        else:
            category = "other_model_associated"
            reason = "stable_model_association_without_prespecified_biological_cross-validation"
        records.append(
            {
                **row._asdict(),
                "is_predeclared_expected_marker": is_expected,
                "is_locked_pam50_gene": is_pam50,
                "supporting_receptors": supporting_receptors,
                "subtype_bh_q_value": subtype_q,
                "evidence_category": category,
                "classification_reason": reason,
                "novelty_caveat": "Potential candidate is not a claim of novelty or causality.",
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    if BIO_DIR.exists() and any(BIO_DIR.iterdir()):
        raise RuntimeError("Biological-validation output directory is not empty")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    verify_inputs(config)
    assignments = pd.read_csv(ASSIGNMENT_PATH, sep="\t")
    samples = pd.read_csv(ANALYSIS_SAMPLE_PATH, sep="\t")
    genes = pd.read_csv(GENE_PATH, sep="\t")
    pam50 = pd.read_csv(PAM50_PATH, sep="\t")
    matrix = np.load(MATRIX_PATH, mmap_mode="r")
    BIO_DIR.mkdir(parents=True, exist_ok=False)

    stable = stable_gene_table(config)
    if stable.empty:
        raise RuntimeError("No genes meet the locked stability rule")
    stable = stable.merge(
        genes[
            ["matrix_column", "gene_id", "gene_id_without_version", "gene_type"]
        ],
        on="matrix_column",
        how="left",
        validate="one_to_one",
    )
    long, distribution, subtype = expression_validation(
        stable, assignments, samples, matrix
    )
    receptor = receptor_validation(long)
    classified = classify_genes(stable, subtype, receptor, pam50, config)
    pam50_overlap = classified[
        [
            "stable_gene_order",
            "matrix_column",
            "gene_name",
            "stable_source",
            "maximum_top20_fold_count",
            "is_predeclared_expected_marker",
            "is_locked_pam50_gene",
            "evidence_category",
        ]
    ].copy()
    source_map = pam50.set_index("current_symbol")["source_symbol"].to_dict()
    pam50_overlap["pam50_source_symbol"] = pam50_overlap["gene_name"].map(source_map)

    stable.to_csv(BIO_DIR / "stable_important_genes.tsv", sep="\t", index=False)
    long.to_csv(
        BIO_DIR / "stable_gene_expression_by_development_sample.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    distribution.to_csv(
        BIO_DIR / "stable_gene_subtype_expression_summary.tsv", sep="\t", index=False
    )
    subtype.to_csv(
        BIO_DIR / "stable_gene_subtype_tests.tsv", sep="\t", index=False
    )
    pam50_overlap.to_csv(
        BIO_DIR / "stable_gene_pam50_overlap.tsv", sep="\t", index=False
    )
    receptor.to_csv(
        BIO_DIR / "stable_gene_receptor_associations.tsv", sep="\t", index=False
    )
    classified.to_csv(
        BIO_DIR / "stable_gene_evidence_classification.tsv", sep="\t", index=False
    )

    background = pd.read_csv(INTERPRET_DIR / "enrichment_background.tsv", sep="\t")
    union = background.loc[background["in_union_background"], "gene_name"].dropna().astype(str)
    consensus = background.loc[
        background["in_all_five_folds_background"], "gene_name"
    ].dropna().astype(str)
    stable_names = classified["gene_name"].astype(str)
    request = {
        "schema_version": "1.0.0",
        "created_at_utc": now_utc(),
        "stable_gene_rule": config["biological_validation"]["stable_gene_rule"],
        "query_gene_count": int(stable_names.nunique()),
        "query_genes": sorted(stable_names.unique()),
        "primary_background_rule": config["biological_validation"]["enrichment_background"],
        "primary_background_gene_count": int(union.nunique()),
        "primary_background_genes": sorted(union.unique()),
        "sensitivity_background_rule": config["biological_validation"][
            "enrichment_background_sensitivity"
        ],
        "sensitivity_background_gene_count": int(consensus.nunique()),
        "sensitivity_background_genes": sorted(consensus.unique()),
        "sources": config["biological_validation"]["enrichment_sources"],
        "domain_scope": "custom",
        "multiple_testing": "fdr",
        "user_threshold": config["biological_validation"]["fdr_threshold"],
    }
    (BIO_DIR / "functional_enrichment_request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    artifact_names = [
        "stable_important_genes.tsv",
        "stable_gene_expression_by_development_sample.tsv.gz",
        "stable_gene_subtype_expression_summary.tsv",
        "stable_gene_subtype_tests.tsv",
        "stable_gene_pam50_overlap.tsv",
        "stable_gene_receptor_associations.tsv",
        "stable_gene_evidence_classification.tsv",
        "functional_enrichment_request.json",
    ]
    manifest = {
        "schema_version": "1.0.0",
        "status": "COMPLETE_AWAITING_ENRICHMENT_API",
        "created_at_utc": now_utc(),
        "development_expression_n": 756,
        "locked_test_expression_rows_loaded": 0,
        "stable_gene_n": len(stable),
        "pam50_overlap_n": int(classified["is_locked_pam50_gene"].sum()),
        "expected_marker_overlap_n": int(
            classified["is_predeclared_expected_marker"].sum()
        ),
        "input_sha256": {
            "config/interpretability_v1.json": sha256(CONFIG_PATH),
            "outputs/interpretability/run_manifest.json": sha256(
                INTERPRET_DIR / "run_manifest.json"
            ),
            "outputs/eda/tables/analysis_samples.tsv": sha256(ANALYSIS_SAMPLE_PATH),
            "data/processed/labels/pam50_signature_genes_v1.tsv": sha256(PAM50_PATH),
        },
        "output_sha256": {name: sha256(BIO_DIR / name) for name in artifact_names},
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    (BIO_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_AWAITING_ENRICHMENT_API",
                "stable_gene_n": len(stable),
                "pam50_overlap_n": int(classified["is_locked_pam50_gene"].sum()),
                "classification_counts": classified["evidence_category"].value_counts().to_dict(),
                "background_union_n": int(union.nunique()),
                "background_all_five_n": int(consensus.nunique()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
