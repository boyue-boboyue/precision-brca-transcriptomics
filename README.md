# OncoStratify-BRCA

Interpretable machine learning for breast cancer molecular subtype classification from TCGA-BRCA transcriptomic data.

## Current status

Snapshot date: **2026-09-11**

- GDC Release 46.0 TCGA-BRCA STAR Counts downloaded and verified: 1,111 files, 1,095 unique Primary Tumor cases.
- PanCancer Atlas PAM50 labels locked: 981 five-class cases and 945 four-class primary-analysis cases.
- Case-level expression matrices built: 1,095 samples × 60,660 genes, including counts, TPM, and log2(TPM + 1).
- Exploratory analysis completed and verified: inclusion flow, class balance, expression/library QC, PCA, batch diagnostics, a 500-HVG clustered heatmap, cluster/subtype correspondence, and ER/PR/HER2 contingency analysis.
- Patient-level evaluation assignments locked with seed `20260909`: the four-class primary cohort contains 756 development and 189 locked-test cases; the five-class sensitivity cohort contains 785 development and 196 locked-test cases.
- Five-by-five nested stratified CV folds, preprocessing Pipelines, candidate models and hyperparameter grids are locked.
- Development-only Dummy, multinomial logistic L2 and elastic-net nested-CV comparison completed. Elastic-net achieved mean macro F1 `0.890` versus `0.866` for L2 while retaining a mean of 171/1,000 non-zero genes. The locked test set has not been evaluated.
- Development-only LinearSVC nested CV completed with mean macro F1 `0.872`; training-only sigmoid calibration used `CalibratedClassifierCV`, and no RBF model was fitted.
- Development-only balanced random-forest nested CV completed with mean macro F1 `0.905`, balanced accuracy `0.905`, and accuracy `0.917`. All five outer folds selected 2,000 training-fold high-variance genes, 750 trees, depth 16, leaf size 1, and `max_features="sqrt"`. It has the highest mean macro F1, but exceeded elastic-net in only 2/5 paired outer folds and had worse log-loss (`0.367` versus `0.268`), so final model selection is not yet frozen.

See [`docs/protocol.md`](docs/protocol.md) for the preregistered-style analysis plan and [`docs/handoff.md`](docs/handoff.md) for the exact continuation point.

## Key directories

```text
data/raw/                    Source downloads; large GDC STAR Counts are not portable by default
data/manifests/              GDC query responses and download manifest
data/metadata/               Case, sample, aliquot, annotation, and verification metadata
data/processed/labels/       Locked PanCancer Atlas PAM50 labels
data/processed/expression/   Expression matrices, axes, cohort indices, and matrix manifest
docs/                        Protocol and continuation handoff
scripts/                     Reproducible data, matrix, and EDA scripts
outputs/eda/                 Verified figures, tables, report, and artifact manifest
outputs/modeling/            Nested-CV metrics, OOF predictions, model reports, and hashes
outputs/exports/             Portable handoff archives
```

## GitHub repository scope

The repository includes source code, locked configurations, query manifests,
metadata, labels, matrix axes, predefined evaluation splits, EDA artifacts, and
development-only modeling outputs. Raw GDC STAR Counts, the three large derived
expression arrays, and local handoff archives are intentionally excluded from
normal Git history. See [`docs/data_availability.md`](docs/data_availability.md)
for exact exclusions and reconstruction instructions.

## Resume from processed data

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-eda.txt
.venv/bin/python scripts/verify_expression_matrix.py
.venv/bin/python scripts/verify_eda.py
.venv/bin/python scripts/verify_evaluation_framework.py
.venv/bin/python scripts/verify_logistic_comparison.py
.venv/bin/python scripts/verify_linear_svc.py
.venv/bin/python scripts/verify_random_forest.py
```

To reproduce the EDA after verification:

```bash
.venv/bin/python scripts/run_eda.py
```

The next modeling step is a clearly labeled development-only sensitivity stage: examine the 2,000-gene random-forest grid boundary and repeat the prespecified PAM50-gene-excluded analysis. Then freeze the final model and decision rule. Do not evaluate the locked test set before that freeze.

## Full rebuild order

```bash
bash scripts/query_gdc.sh
bash scripts/download_gdc_star_counts.sh
bash scripts/verify_gdc_download.sh
bash scripts/build_gdc_metadata_tables.sh
bash scripts/fetch_pancanatlas_pam50.sh
python3 scripts/lock_pancanatlas_pam50.py
python3 scripts/build_expression_matrix.py
python3 scripts/verify_expression_matrix.py
bash scripts/fetch_cbioportal_receptors.sh
python3 scripts/run_eda.py
python3 scripts/verify_eda.py
python3 scripts/lock_evaluation_splits.py
python3 scripts/verify_evaluation_framework.py
```

Network access is required only for the GDC/cBioPortal fetch steps.
