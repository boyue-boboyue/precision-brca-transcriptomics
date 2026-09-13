# Processed expression matrix release

## Permanent release

- Release: [`data-v1.0.0`](https://github.com/boyue-boboyue/precision-brca-transcriptomics/releases/tag/data-v1.0.0)
- Asset: [`oncostratify-brca-expression-matrices-data-v1.0.0.tar.gz`](https://github.com/boyue-boboyue/precision-brca-transcriptomics/releases/download/data-v1.0.0/oncostratify-brca-expression-matrices-data-v1.0.0.tar.gz)
- Size: 354,035,355 bytes (337.63 MiB)
- SHA-256: `fb800d92bd7c958aa854204dbc87304943c72231885c50e4fa6468054d079383`

The archive is the formal distribution of the Git-excluded processed matrices
used by the canonical analysis. It contains only
`data/processed/expression/` and `data/processed/labels/`; it does not include
raw GDC files, credentials, controlled-access sequence data, model outputs, or
local environments.

## Contents

| Matrix | Shape | dtype | Definition | SHA-256 |
|---|---:|---|---|---|
| `counts_uint32.npy` | 1,095 × 60,660 | `uint32` | GDC STAR unstranded counts | `8a820d8dd77a47ec8b133d7221bd9d36a95a7b338c49d1ab97589a9b06f10eb4` |
| `tpm_float32.npy` | 1,095 × 60,660 | `float32` | GDC STAR unstranded TPM | `6e7a19af690dccb941bebf06acccc1429861918417625a39f8e4ab5fee1273c8` |
| `log2_tpm_float32.npy` | 1,095 × 60,660 | `float32` | `log2(TPM + 1)` | `992cf12149d7e2bf1e0185a90560835bc9643e89397c2a1200ee4d192eef3542` |

The same directory includes ordered sample/gene axes, protein-coding and PAM50
cohort indices, candidate-file selection audit, matrix manifest, verification
report, locked PanCancer Atlas labels, and the PAM50 signature table. Rows are
samples and columns are genes. The complete machine-readable release record is
[`processed_matrices_data_v1.0.0.json`](https://github.com/boyue-boboyue/precision-brca-transcriptomics/blob/data-v1.0.0/data/manifests/processed_matrices_data_v1.0.0.json).

## Download, verify, and restore

```bash
curl --fail --location --remote-name \
  https://github.com/boyue-boboyue/precision-brca-transcriptomics/releases/download/data-v1.0.0/oncostratify-brca-expression-matrices-data-v1.0.0.tar.gz
```

Verify on Linux:

```bash
sha256sum oncostratify-brca-expression-matrices-data-v1.0.0.tar.gz
```

Verify on macOS:

```bash
shasum -a 256 oncostratify-brca-expression-matrices-data-v1.0.0.tar.gz
```

The observed digest must equal the SHA-256 above. Restore the files at the root
of a repository clone and run the committed matrix verifier:

```bash
tar -xzf oncostratify-brca-expression-matrices-data-v1.0.0.tar.gz
make environment
.venv/bin/python scripts/verify_expression_matrix.py
```

Extraction restores files under `data/processed/expression/` and
`data/processed/labels/`. It does not overwrite the model, split, final-test, or
interpretability lock files.

## Provenance and reuse boundary

The matrices derive from 1,111 open-access TCGA-BRCA STAR Counts files locked in
the repository GDC manifest. One Primary Tumor sample was selected per case,
yielding 1,095 rows. The source is NCI GDC Data Release 46.0 (August 10, 2026),
and the gene model is GENCODE v36.

Users must follow applicable NCI GDC and TCGA data-use and attribution policies.
The release is intended for research reproducibility, not clinical diagnosis,
prognosis, or treatment selection. TCGA case barcodes are public study
identifiers and are not direct personal identifiers.
