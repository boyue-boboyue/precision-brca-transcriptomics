#!/usr/bin/env python3
"""Query g:Profiler GO:BP/Reactome enrichment with locked custom backgrounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BIO_DIR = ROOT / "outputs" / "interpretability" / "biological_validation"
REQUEST_PATH = BIO_DIR / "functional_enrichment_request.json"
MANIFEST_PATH = BIO_DIR / "run_manifest.json"
PROFILE_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
VERSION_URL = "https://biit.cs.ut.ee/gprofiler/api/util/data_versions/?organism=hsapiens"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "OncoStratify-BRCA/1.0 reproducible-research",
        },
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def flatten_results(response: dict[str, Any], analysis: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for result in response.get("result", []):
        intersections = result.get("intersections", [])
        query_name = result.get("query", "query_1")
        query_genes = response.get("meta", {}).get("query_metadata", {}).get(
            "queries", {}
        ).get(query_name, [])
        if len(intersections) != len(query_genes):
            raise RuntimeError(
                f"g:Profiler intersections/query length mismatch for {query_name}"
            )
        intersection_genes = [
            str(gene)
            for gene, evidence in zip(query_genes, intersections)
            if evidence
        ]
        evidence_codes = sorted(
            {
                str(code)
                for evidence in intersections
                for code in (evidence if isinstance(evidence, list) else [evidence])
                if code
            }
        )
        if len(intersection_genes) != int(result.get("intersection_size", 0)):
            raise RuntimeError("Reconstructed intersection size does not match API result")
        records.append(
            {
                "analysis": analysis,
                "source": result.get("source"),
                "term_id": result.get("native"),
                "term_name": result.get("name"),
                "adjusted_p_value": result.get("p_value"),
                "significant": result.get("significant"),
                "query_size": result.get("query_size"),
                "term_size": result.get("term_size"),
                "intersection_size": result.get("intersection_size"),
                "effective_domain_size": result.get("effective_domain_size"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "intersection_genes": ";".join(intersection_genes),
                "intersection_evidence_codes": ";".join(evidence_codes),
            }
        )
    columns = [
        "analysis",
        "source",
        "term_id",
        "term_name",
        "adjusted_p_value",
        "significant",
        "query_size",
        "term_size",
        "intersection_size",
        "effective_domain_size",
        "precision",
        "recall",
        "intersection_genes",
        "intersection_evidence_codes",
    ]
    return pd.DataFrame(records, columns=columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild-table-from-raw",
        action="store_true",
        help="Rebuild only the tabular export from the exact saved raw responses.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not REQUEST_PATH.exists():
        raise RuntimeError("Run scripts/run_biological_validation.py first")
    if (BIO_DIR / "functional_enrichment_results.tsv").exists() and not args.rebuild_table_from_raw:
        raise RuntimeError("Refusing to overwrite completed enrichment results")
    locked = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    base = {
        "organism": "hsapiens",
        "query": locked["query_genes"],
        "sources": locked["sources"],
        "user_threshold": locked["user_threshold"],
        "significance_threshold_method": "fdr",
        "domain_scope": locked["domain_scope"],
        "no_iea": False,
        "no_evidences": False,
        "ordered": False,
    }
    raw_paths = {
        "versions": BIO_DIR / "gprofiler_data_versions.json",
        "primary": BIO_DIR / "gprofiler_primary_raw.json",
        "sensitivity": BIO_DIR / "gprofiler_consensus_background_raw.json",
    }
    if all(path.exists() for path in raw_paths.values()):
        print(f"[{now_utc()}] Resuming from exact saved g:Profiler raw responses", flush=True)
        versions = json.loads(raw_paths["versions"].read_text(encoding="utf-8"))
        primary = json.loads(raw_paths["primary"].read_text(encoding="utf-8"))
        sensitivity = json.loads(raw_paths["sensitivity"].read_text(encoding="utf-8"))
        if args.rebuild_table_from_raw:
            print(
                f"[{now_utc()}] Corrective table rebuild; no API request will be made",
                flush=True,
            )
    elif any(path.exists() for path in raw_paths.values()):
        raise RuntimeError("Partial raw enrichment response set; manual audit required")
    else:
        print(f"[{now_utc()}] Requesting g:Profiler data versions", flush=True)
        versions = request_json(VERSION_URL)
        print(f"[{now_utc()}] GO:BP/Reactome enrichment with union background", flush=True)
        primary = request_json(
            PROFILE_URL, {**base, "background": locked["primary_background_genes"]}
        )
        print(f"[{now_utc()}] Background sensitivity enrichment (5/5-fold pass)", flush=True)
        sensitivity = request_json(
            PROFILE_URL, {**base, "background": locked["sensitivity_background_genes"]}
        )
        raw_paths["versions"].write_text(
            json.dumps(versions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raw_paths["primary"].write_text(
            json.dumps(primary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raw_paths["sensitivity"].write_text(
            json.dumps(sensitivity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    table = pd.concat(
        [
            flatten_results(primary, "primary_union_background"),
            flatten_results(sensitivity, "sensitivity_all_five_folds_background"),
        ],
        ignore_index=True,
    )
    table = table.sort_values(
        ["analysis", "adjusted_p_value", "source", "term_id"], na_position="last"
    )
    table.to_csv(BIO_DIR / "functional_enrichment_results.tsv", sep="\t", index=False)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "COMPLETE",
            "enrichment_completed_at_utc": now_utc(),
            "enrichment_service": PROFILE_URL,
            "enrichment_sources": locked["sources"],
            "enrichment_domain_scope": "custom",
            "enrichment_multiple_testing": "Benjamini-Hochberg FDR",
            "enrichment_primary_result_n": int(
                table["analysis"].eq("primary_union_background").sum()
            ),
            "enrichment_sensitivity_result_n": int(
                table["analysis"].eq("sensitivity_all_five_folds_background").sum()
            ),
            "enrichment_fetcher_sha256": sha256(Path(__file__).resolve()),
        }
    )
    manifest["output_sha256"].update(
        {
            name: sha256(BIO_DIR / name)
            for name in [
                "functional_enrichment_results.tsv",
                "gprofiler_primary_raw.json",
                "gprofiler_consensus_background_raw.json",
                "gprofiler_data_versions.json",
            ]
        }
    )
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "primary_terms": manifest["enrichment_primary_result_n"],
                "sensitivity_terms": manifest["enrichment_sensitivity_result_n"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
