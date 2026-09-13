# Data availability and GitHub scope

This repository versions the code, locked analysis configuration, data manifests,
metadata, labels, matrix axes, evaluation assignments, exploratory figures, and
development-set modeling results required to audit the project.

## Files intentionally excluded from GitHub

| Local artifact | Approximate size | Reason |
|---|---:|---|
| `data/raw/gdc/star_counts/` | 4.4 GiB | Re-downloadable open-access GDC source files |
| `data/raw/gdc/publication_supplement/` | 332 KiB | Git-excluded third-party TCGA 2012 originals; the analysis-ready TSV is versioned |
| Three `data/processed/expression/*.npy` matrices | 765 MiB total | Published in the versioned [`data-v1.0.0` GitHub Release](https://github.com/boyue-boboyue/precision-brca-transcriptomics/releases/tag/data-v1.0.0) |
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

The default cohort verifier treats the two TCGA 2012 publication-supplement
originals as one optional source unit: it verifies their recorded SHA-256
digests when both are present, skips the unit when both are absent from a fresh
clone, and rejects a partial restore. The versioned analysis-ready TSV and all
canonical robustness outputs remain independently hash-checked. To require the
original files explicitly, restore both paths listed in
`data/metadata/brca_2012_supplement.sha256` and run:

```bash
.venv/bin/python scripts/verify_source_artifacts.py --stage cohort
```

### Linux and macOS download verification

The download and verification entry points use a Python standard-library
validator for identical behaviour on Linux and macOS:

```bash
PYTHON=python3 bash scripts/verify_gdc_download.sh
```

The validator checks the locked manifest schema, UUIDs, released state, safe
target paths, expected byte sizes, and MD5 digests. Verification fails for a
missing or corrupt manifest member, an unexpected file, or a residual `.part`
download:

```text
manifest schema → safe target path → exact size → MD5 → complete file inventory
```

Reports are written atomically to
`data/metadata/gdc_download_verification.tsv` and
`data/metadata/gdc_download_verification_summary.json`. Validation does not
require platform-specific `stat` flags or external `md5`, `md5sum`, `jq`, or
`shasum` commands. A small fixture test runs on Ubuntu in GitHub Actions; it
does not download or expose genomic data.

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
also versioned. The locked test was evaluated exactly once after the final model
and procedure were frozen; the access record and resulting artifacts are
versioned.

## Formal processed-matrix release

The matrices, axes, cohort indices, and locked label tables are available as a
single versioned release asset:

- [Release page](https://github.com/boyue-boboyue/precision-brca-transcriptomics/releases/tag/data-v1.0.0)
- [Direct archive download](https://github.com/boyue-boboyue/precision-brca-transcriptomics/releases/download/data-v1.0.0/oncostratify-brca-expression-matrices-data-v1.0.0.tar.gz)
- SHA-256: `fb800d92bd7c958aa854204dbc87304943c72231885c50e4fa6468054d079383`
- [Contents, provenance, and restoration instructions](processed_matrix_release.md)

The asset contains only `data/processed/expression/` and
`data/processed/labels/`. It was validated against every hash in
`matrix_manifest.json` before publication.

## Local handoff archives

The local `outputs/exports/` directory contains core and processed-data archives.
The archives themselves are ignored by Git, while `archive_checksums.sha256` is
retained as an audit record. These dated local bundles are superseded for data
distribution by the formal `data-v1.0.0` release and should not be committed to
normal Git history.
