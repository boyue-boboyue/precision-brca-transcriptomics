# OncoStratify-BRCA

Interpretable machine learning for breast cancer molecular subtype classification from TCGA-BRCA transcriptomic data.

## Current status

Snapshot date: **2026-09-13**

- GDC Release 46.0 TCGA-BRCA STAR Counts downloaded and verified: 1,111 files, 1,095 unique Primary Tumor cases.
- PanCancer Atlas PAM50 labels locked: 981 five-class cases and 945 four-class primary-analysis cases.
- Case-level expression matrices built: 1,095 samples × 60,660 genes, including counts, TPM, and log2(TPM + 1).
- Exploratory analysis completed and verified: inclusion flow, class balance, expression/library QC, PCA, batch diagnostics, a 500-HVG clustered heatmap, cluster/subtype correspondence, and ER/PR/HER2 contingency analysis.
- Patient-level evaluation assignments locked with seed `20260909`: the four-class primary cohort contains 756 development and 189 locked-test cases; the five-class sensitivity cohort contains 785 development and 196 locked-test cases.
- Five-by-five nested stratified CV folds, preprocessing Pipelines, candidate models and hyperparameter grids are locked.
- Development-only Dummy, multinomial logistic L2 and elastic-net nested-CV comparison completed. Elastic-net achieved mean macro F1 `0.890` versus `0.866` for L2 while retaining a mean of 171/1,000 non-zero genes.
- Development-only LinearSVC nested CV completed with mean macro F1 `0.872`; training-only sigmoid calibration used `CalibratedClassifierCV`, and no RBF model was fitted.
- Development-only balanced random-forest nested CV completed with mean macro F1 `0.905`, balanced accuracy `0.905`, and accuracy `0.917`. All five outer folds selected 2,000 training-fold high-variance genes, 750 trees, depth 16, leaf size 1, and `max_features="sqrt"`.
- The 50-gene PAM50 signature was locked and removed before every Pipeline step for a full five-model sensitivity rerun. Random-forest performance was nearly unchanged: macro F1 `0.902` excluded versus `0.905` included; balanced accuracy `0.906` versus `0.905`.
- The preregistered mean outer-fold macro F1 rule selected the PAM50-included random forest; its `0.015` lead over elastic-net exceeded the `0.01` tie threshold. The final model and procedure are frozen in `config/final_model_lock_v1.json`.
- The 189-case locked test was evaluated exactly once. Final random-forest balanced accuracy was `0.907` (95% stratified-bootstrap CI `0.855–0.945`) and macro F1 was `0.885` (`0.836–0.929`).
- Three-level interpretability is complete and verified: cross-fold Elastic-net coefficients, random-forest permutation importance computed only on the five outer validation folds, and TreeSHAP for six prediction-defined locked-test cases using a 100-case development-only background.
- The locked top-20 stability rule identified 58 genes; 14 overlap the locked PAM50 signature and four overlap the predeclared marker audit list. Development-only subtype distributions, PAM50 overlap, ER/PR/HER2 IHC associations, and GO:BP/Reactome enrichment with a 15,238-gene expression-filter background are available in the integrated report.
- Post-selection robustness analysis is complete across 13 fixed-model scenarios (65 outer fits) without reusing the locked test. Four-class macro F1 was stable to PAM50-gene exclusion (`0.899`), log2 CPM (`0.911`), three low-expression rules (`0.906–0.909`), and three grouped seeds (`0.905–0.907`); removing class weights reduced it to `0.878`. Five-class macro F1 was `0.817`, driven partly by the 29-case Normal-like class (pooled F1 `0.512`).
- Label propagation from the raw PanCancer Atlas response through the expression and split tables is 100% concordant. The independent TCGA 2012 PAM50 freeze agrees with PanCancer Atlas for 398/447 overlapping cases (`89.0%`, Cohen's κ `0.838`). The complete limitations analysis and reproducibility audit both pass.

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
outputs/final_evaluation/     One-time locked-test predictions, metrics, curves, and bootstrap CIs
outputs/performance_report/   Unified tables, figures, model selection, report, and verification
outputs/interpretability/     Global/class/individual explanations, biological validation, figures, and verification
outputs/robustness/           Fixed-model sensitivity analyses, label audit, limitations report, and verification
outputs/exports/             Portable handoff archives
```

## GitHub repository scope

The repository includes source code, locked configurations, query manifests,
metadata, labels, matrix axes, predefined evaluation splits, EDA artifacts, and
development-only modeling outputs. Raw GDC STAR Counts, the three large derived
expression arrays, and local handoff archives are intentionally excluded from
normal Git history. See [`docs/data_availability.md`](docs/data_availability.md)
for exact exclusions and reconstruction instructions.

## Reproducible Make entry points

```bash
make environment
make metadata
make cohort
make matrix
make eda
make train
make evaluate
make explain
make test
```

The targets form one explicit chain:

```text
environment → metadata → cohort → matrix → eda → train → evaluate → explain → test
```

Calling a downstream target automatically runs its prerequisites, so a bare
`make` or `make test` checks the complete delivery. `make matrix` verifies the
three local arrays when present; in a fresh clone it downloads the 1,111 files
in the locked GDC manifest, verifies MD5 and byte sizes, rebuilds the arrays,
and requires their hashes to match the committed matrix manifest. Download
parallelism can be changed, for example with `GDC_DOWNLOAD_WORKERS=4`.

The final test and the six SHAP test cases are protected by one-time access
records. For that reason `train`, `evaluate`, and `explain` verify the frozen
canonical result chain without rerunning those irreversible steps. Tables and
figures that do not reopen the locked test can be redrawn from frozen result
tables with:

```bash
make test REBUILD_REPORTS=1
```

Model interpretation, internal biological validation, and the specified robustness analyses are complete. The next scientific priority is external-cohort validation with frozen preprocessing, genes, parameters, and label mapping. Any feature-count expansion beyond the preregistered 2,000-gene grid must remain a clearly labeled post-result development-only sensitivity analysis and cannot trigger another locked-test evaluation.

Network access is required only when the Git-omitted GDC STAR Counts must be
downloaded. Metadata, PAM50 annotations, receptor sources, and enrichment raw
responses are version-frozen and checksum-verified locally by default.
