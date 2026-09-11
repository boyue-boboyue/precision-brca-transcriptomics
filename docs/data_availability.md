# Data availability and GitHub scope

This repository versions the code, locked analysis configuration, data manifests,
metadata, labels, matrix axes, evaluation assignments, exploratory figures, and
development-set modeling results required to audit the project.

## Files intentionally excluded from GitHub

| Local artifact | Approximate size | Reason |
|---|---:|---|
| `data/raw/gdc/star_counts/` | 4.4 GiB | Re-downloadable open-access GDC source files |
| Three `data/processed/expression/*.npy` matrices | 765 MiB total | Derived arrays exceed ordinary GitHub file limits |
| `outputs/exports/*.tar.gz`, `outputs/handoff/` | 369 MiB total | Local dated handoff bundle and its generated metadata |
| `.venv/`, `logs/`, `work/` | environment-dependent | Reproducible or temporary files |

The exclusions are implemented in `.gitignore`. No controlled-access genomic
data, credentials, or direct identifiers are included.

## Reconstructing the source data

The exact GDC file inventory is stored in
`data/manifests/gdc_manifest_tcga_brca_star_counts.tsv`. Query payloads,
retrieval metadata, file sizes, MD5 checks, and cohort summaries are versioned
alongside it.

```bash
bash scripts/query_gdc.sh
bash scripts/download_gdc_star_counts.sh
bash scripts/verify_gdc_download.sh
bash scripts/build_gdc_metadata_tables.sh
```

The downloader resumes partial files and validates both byte size and MD5 before
promoting each download.

## Reconstructing the expression matrices

After restoring the raw STAR Counts files:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-expression.txt
.venv/bin/python scripts/build_expression_matrix.py
.venv/bin/python scripts/verify_expression_matrix.py
```

`data/processed/expression/matrix_manifest.json` records the expected shape,
selection rules, source hashes, and SHA-256 values of the three derived arrays.
The versioned sample and gene axes allow the matrix definition to be audited
without storing the arrays in Git.

## Labels and evaluation assignments

The locked PanCancer Atlas PAM50 source responses and derived labels are included
because they are small and required to reproduce cohort membership. Patient-level
development/locked-test assignments and all nested cross-validation folds are
also versioned. The locked test set has not been evaluated.

## Local handoff archives

The local `outputs/exports/` directory contains core and processed-data archives.
The archives themselves are ignored by Git, while `archive_checksums.sha256` is
retained as an audit record. A future data release should use an appropriate
research-data repository or a GitHub Release backed by external storage rather
than committing these binaries to normal Git history.
