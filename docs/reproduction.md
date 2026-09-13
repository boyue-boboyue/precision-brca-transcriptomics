# Isolated reproduction mode

## Purpose

Reproduction mode computationally replays the frozen analysis without
rewriting the canonical final-model lock, split lock, PAM50 lock, final-test
access record, interpretability lock, or SHAP access record. The published
repository remains the reference analysis; every replay receives its own
working tree, locks, access records, outputs, and audit trail.

## Run the complete replay

Use a new directory name for each independent replay:

```bash
make test \
  REPRODUCTION=1 \
  REPRO_DIR=work/reproduction/run-001 \
  N_JOBS=1 \
  FOREST_N_JOBS=4 \
  GDC_DOWNLOAD_WORKERS=12
```

The normal Make dependency chain is retained:

```text
environment → metadata → cohort → matrix → eda → train → evaluate → explain → test
```

To inspect every command without creating or changing any file:

```bash
make test REPRODUCTION=1 REPRO_DRY_RUN=1
```

To stop at, or resume from, an intermediate stage, use the same `REPRO_DIR`:

```bash
make train REPRODUCTION=1 REPRO_DIR=work/reproduction/run-001
make evaluate REPRODUCTION=1 REPRO_DIR=work/reproduction/run-001
make test REPRODUCTION=1 REPRO_DIR=work/reproduction/run-001
```

Completed stages are verified again and are not recomputed. The coordinator
never deletes or overwrites a partial stage. If execution stops after a stage
has begun but before its completion marker is written, retain that directory
for audit and restart with a new `REPRO_DIR`.

## Isolation contract

The coordinator creates a source snapshot that excludes:

- `.git`, `.venv`, logs, previous work directories, and canonical outputs;
- the canonical `final_model_lock_v1.json` and `interpretability_v1.json`;
- canonical final-test and interpretation access records;
- canonical SHAP case/background selections;
- the three large expression arrays and raw GDC STAR Counts.

Existing large expression arrays and raw STAR Counts are linked from the source
project and treated as read-only to avoid duplicating several gigabytes. The
matrix coordinator never downloads through a linked raw-data directory. If the
expression arrays are unavailable, the isolated `matrix` stage reconstructs
them from the locked GDC manifest inside the reproduction directory.

Before and after every stage, the coordinator hashes all canonical JSON
configurations, split artifacts, label locks, PAM50 signature, and SHAP
selection records. Any source-repository lock change aborts the replay. The
source-code/configuration snapshot hash must also remain stable when resuming a
run; code changes require a new reproduction directory.

## Replayed stages

| Stage | Reproduction action |
|---|---|
| `metadata` | Verify frozen GDC query and derived metadata hashes. |
| `cohort` | Verify PanCancer Atlas, receptor, and PAM50 lock provenance; verify the optional TCGA 2012 publication originals when complete, skip them when fully absent, and reject a partial restore. |
| `matrix` | Verify linked arrays or reconstruct and hash-check omitted arrays. |
| `eda` | Rebuild all EDA tables, figures, and the EDA manifest. |
| `train` | Rerun Dummy, logistic L2/elastic-net, LinearSVC, random forest, and PAM50-excluded nested CV. |
| `evaluate` | Create an independent final-model lock, fit the final model, make one isolated test evaluation, and rebuild the unified report. |
| `explain` | Create an independent interpretability lock, recompute permutation importance and SHAP, validate biology, and rerun robustness analyses. |
| `test` | Run all unit/integrity tests and compare key scientific results with the canonical snapshot. |

The enrichment stage reuses the three frozen raw g:Profiler responses only
when the reproduced query genes, backgrounds, sources, threshold, and analysis
options match the canonical request after excluding its timestamp. A changed
request is rejected instead of being silently paired with unrelated responses.

## Audit records

Each reproduction directory contains:

```text
.reproduction/manifest.json
.reproduction/stages/metadata.json
.reproduction/stages/cohort.json
.reproduction/stages/matrix.json
.reproduction/stages/eda.json
.reproduction/stages/train.json
.reproduction/stages/evaluate.json
.reproduction/stages/explain.json
.reproduction/stages/test.json
.reproduction/canonical_equivalence.json
```

The manifest records the source commit, source-tree hash, environment versions,
linked inputs, and initial canonical lock hashes. Every completed stage records
its exact command list and confirms that canonical locks remained unchanged.
The final equivalence report compares key EDA tables, model metrics and OOF
predictions, locked-test metrics, interpretation tables, enrichment results,
and robustness summaries with the published result snapshot. Exact SHA-256
matches are preferred; otherwise only numeric serialization differences within
relative tolerance `1e-9` and absolute tolerance `1e-10` are accepted. Text,
row order, columns, labels, predictions, and table dimensions must still match.

## Scope and runtime

The full replay includes repeated nested cross-validation, permutation
importance, TreeSHAP, and 65 robustness fits. It is intentionally much more
expensive than the default `make test`, which verifies the published frozen
artifacts. Reproduction results remain under the Git-ignored `work/` directory
unless an external `REPRO_DIR` is supplied.
