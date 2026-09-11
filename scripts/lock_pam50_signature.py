#!/usr/bin/env python3
"""Lock the 50-gene PAM50 signature to the expression-matrix gene axis."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GENES_PATH = ROOT / "data" / "processed" / "expression" / "genes.tsv"
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "labels" / "pam50_signature_genes_v1.tsv"
)
CONFIG_PATH = ROOT / "config" / "pam50_exclusion_v1.json"

SOURCE_COMMIT = "9c9b66d1ef22cbfda1df75b626d2dbc68fb9b25c"
SOURCE_URL = (
    "https://raw.githubusercontent.com/bhklab/genefu/"
    f"{SOURCE_COMMIT}/inst/extdata/pam50_model.csv"
)
SOURCE_ROWS = [
    ("ACTR3B", 57180), ("ANLN", 54443), ("BAG1", 573), ("BCL2", 596),
    ("BIRC5", 332), ("BLVRA", 644), ("CCNB1", 891), ("CCNE1", 898),
    ("CDC20", 991), ("CDC6", 990), ("CDCA1", 83540), ("CDH3", 1001),
    ("CENPF", 1063), ("CEP55", 55165), ("CXXC5", 51523), ("EGFR", 1956),
    ("ERBB2", 2064), ("ESR1", 2099), ("EXO1", 9156), ("FGFR4", 2264),
    ("FOXA1", 3169), ("FOXC1", 2296), ("GPR160", 26996), ("GRB7", 2886),
    ("KIF2C", 11004), ("KNTC2", 10403), ("KRT14", 3861), ("KRT17", 3872),
    ("KRT5", 3852), ("MAPT", 4137), ("MDM2", 4193), ("MELK", 9833),
    ("MIA", 8190), ("MKI67", 4288), ("MLPH", 79083), ("MMP11", 4320),
    ("MYBL2", 4605), ("MYC", 4609), ("NAT1", 9), ("ORC6L", 23594),
    ("PGR", 5241), ("PHGDH", 26227), ("PTTG1", 9232), ("RRM2", 6241),
    ("SFRP1", 6422), ("SLC39A6", 25800), ("TMEM45B", 120224),
    ("TYMS", 7298), ("UBE2C", 11065), ("UBE2T", 29089),
]

# HGNC-approved successor symbols represented on the current GENCODE matrix axis.
ALIASES = {"CDCA1": "NUF2", "KNTC2": "NDC80", "ORC6L": "ORC6"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    genes = pd.read_csv(GENES_PATH, sep="\t")
    records: list[dict[str, object]] = []
    for source_order, (source_symbol, entrez_id) in enumerate(SOURCE_ROWS, start=1):
        current_symbol = ALIASES.get(source_symbol, source_symbol)
        matched = genes.loc[
            genes["gene_name"].eq(current_symbol)
            & genes["is_protein_coding"].astype(bool)
        ]
        if len(matched) != 1:
            raise RuntimeError(
                f"Expected one protein-coding match for {source_symbol} -> "
                f"{current_symbol}; observed {len(matched)}"
            )
        gene = matched.iloc[0]
        records.append(
            {
                "source_order": source_order,
                "source_symbol": source_symbol,
                "current_symbol": current_symbol,
                "entrez_gene_id": entrez_id,
                "matrix_column": int(gene["matrix_column"]),
                "gene_id": gene["gene_id"],
                "gene_id_without_version": gene["gene_id_without_version"],
                "gene_type": gene["gene_type"],
                "alias_resolution": (
                    f"{source_symbol}->{current_symbol}"
                    if source_symbol != current_symbol
                    else "direct_symbol_match"
                ),
            }
        )

    locked = pd.DataFrame(records)
    if len(locked) != 50 or locked["matrix_column"].nunique() != 50:
        raise RuntimeError("PAM50 lock must resolve to exactly 50 unique matrix columns")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    locked.to_csv(OUTPUT_PATH, sep="\t", index=False)

    locked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    config = {
        "schema_version": "1.0.0",
        "locked_at_utc": locked_at,
        "analysis": "PAM50-included versus PAM50-excluded development-set sensitivity",
        "source": {
            "name": "genefu PAM50 official centroid model",
            "repository": "https://github.com/bhklab/genefu",
            "commit": SOURCE_COMMIT,
            "raw_csv_url": SOURCE_URL,
            "original_publication": {
                "citation": "Parker et al., J Clin Oncol 2009",
                "doi": "10.1200/JCO.2008.18.1370",
                "pubmed": "https://pubmed.ncbi.nlm.nih.gov/19204204/",
            },
        },
        "gene_count": 50,
        "source_symbol_aliases": ALIASES,
        "exclusion_rule": (
            "Remove the 50 locked matrix columns from the protein-coding candidate "
            "axis before the low-expression filter and all feature selection."
        ),
        "comparison_scope": "development set only; same locked 5x5 folds and model grids",
        "selection_policy": (
            "PAM50-included is the primary model-selection analysis; excluded results "
            "are a prespecified sensitivity analysis and do not access the locked test set."
        ),
        "inputs_sha256": {
            "data/processed/expression/genes.tsv": sha256(GENES_PATH),
            "config/evaluation.json": sha256(ROOT / "config" / "evaluation.json"),
            "config/logistic_comparison_v1.json": sha256(
                ROOT / "config" / "logistic_comparison_v1.json"
            ),
            "config/linear_svc_v1.json": sha256(
                ROOT / "config" / "linear_svc_v1.json"
            ),
            "config/random_forest_v1.json": sha256(
                ROOT / "config" / "random_forest_v1.json"
            ),
            "data/processed/splits/split_lock.json": sha256(
                ROOT / "data" / "processed" / "splits" / "split_lock.json"
            ),
        },
        "locked_gene_table": str(OUTPUT_PATH.relative_to(ROOT)),
        "locked_gene_table_sha256": sha256(OUTPUT_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
