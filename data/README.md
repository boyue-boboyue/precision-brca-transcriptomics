# TCGA-BRCA data inventory

Retrieved from the NCI Genomic Data Commons on 2026-09-09.

## GDC release

- Data release: **46.0**
- Release date: **2026-08-10**
- API version: **1**
- GDC software tag: **8.5.0**
- Status response: `metadata/gdc_status.json`

## RNA-seq data

The saved cohort query selects:

- project: `TCGA-BRCA`
- data category: `Transcriptome Profiling`
- data type: `Gene Expression Quantification`
- experimental strategy: `RNA-Seq`
- workflow: `STAR - Counts`
- access: `open`
- sample type: `Primary Tumor`

Retrieved inventory:

- files: **1,111**
- unique cases represented: **1,095**
- bytes: **4,713,913,640**
- GiB: **4.3901741877**
- files passing size and MD5 verification: **1,111/1,111**
- failed files: **0**

Raw STAR Counts files are stored at:

```text
raw/gdc/star_counts/<file_uuid>/<file_name>
```

The files are intentionally excluded from Git because they occupy approximately
4.4 GiB. The versioned manifest is sufficient to retrieve the same GDC file
objects while they remain available.

## Metadata

| File | Contents |
|---|---|
| `metadata/gdc_star_counts_files.json` | Complete GDC files API response |
| `metadata/gdc_star_counts_file_sample_map.tsv` | File-to-case/sample/aliquot mapping |
| `metadata/gdc_cases.json` | Expanded metadata for all 1,098 TCGA-BRCA cases |
| `metadata/gdc_cases_clinical.tsv` | One-row-per-case clinical summary |
| `metadata/gdc_samples.tsv` | Sample-level biospecimen metadata |
| `metadata/gdc_aliquots.tsv` | Portion, analyte, and aliquot metadata |
| `metadata/gdc_annotations.json` | Complete TCGA-BRCA GDC annotation response |
| `metadata/gdc_annotations.tsv` | Flattened 6,934-row annotation table |
| `metadata/gdc_download_verification.tsv` | Per-file size and MD5 verification |
| `metadata/gdc_download_verification_summary.json` | Verification summary |

GDC annotations cover multiple data types. A `General` notification is not by
itself a reason to exclude an RNA-seq sample. Exclusions must consider entity
type, classification, category, notes, and whether the annotation affects the
selected RNA-seq file, sample, aliquot, or case.

## TCGA-BRCA publication supplement

The GDC-hosted 2012 TCGA-BRCA final sample summary is stored in:

```text
raw/gdc/publication_supplement/
```

The tab-delimited source uses carriage-return line endings. A line-normalized
copy is stored as `metadata/brca_2012_final_full_sample_summary.tsv`.

The historical table contains 463 tumour samples with a non-empty PAM50 label:

| PAM50 label | Samples |
|---|---:|
| Basal | 81 |
| Her2 | 53 |
| LumA | 209 |
| LumB | 112 |
| Normal | 8 |

This supplement represents the publication's 2011 data freeze. It is retained
for historical consistency checks and is not the project's primary label source.

## Locked PanCancer Atlas PAM50 annotations

The primary label source is the patient-level `SUBTYPE` clinical attribute from
cBioPortal study `brca_tcga_pan_can_atlas_2018` (Breast Invasive Carcinoma,
TCGA PanCancer Atlas). The source was retrieved through the cBioPortal REST API
on 2026-09-09 and locked before model fitting.

Locked inventory:

- unique labelled cases: **981**
- labelled cases matched to current GDC STAR Counts: **981**
- GDC STAR Counts cases without a PanCancer Atlas label: **114**
- conflicting patient labels: **0**
- duplicate source records: **0**
- four-class primary cohort before subsequent QC: **945**
- five-class sensitivity cohort before subsequent QC: **981**

| Standard label | Cases |
|---|---:|
| Luminal A | 499 |
| Luminal B | 197 |
| Basal-like | 171 |
| HER2-enriched | 78 |
| Normal-like | 36 |

The immutable label artifacts are:

- `processed/labels/pancanatlas_pam50_locked.tsv`
- `processed/labels/pancanatlas_pam50_gdc_matched.tsv`
- `processed/labels/gdc_star_counts_cases_without_pam50.tsv`
- `processed/labels/pancanatlas_pam50_lock.json`

The lock JSON records the retrieval timestamp, study import date, class counts,
GDC matching counts, and SHA-256 digests for the source response, GDC mapping,
and generated tables. Raw paginated API responses are retained under
`raw/pancanatlas/cbioportal/brca_tcga_pan_can_atlas_2018/` and are explicitly
allowlisted by `.gitignore` for future version control despite the general
raw-data ignore rule.

## Reproducible queries

Saved API payloads:

- `manifests/gdc_star_counts_query.json`
- `manifests/gdc_cases_query.json`
- `manifests/gdc_annotations_query.json`

Saved download manifest:

- `manifests/gdc_manifest_tcga_brca_star_counts.tsv`

To repeat the GDC queries and regenerate metadata tables:

```bash
bash scripts/query_gdc.sh
bash scripts/build_gdc_metadata_tables.sh
```

To download or resume the RNA-seq files and verify them:

```bash
bash scripts/download_gdc_star_counts.sh
bash scripts/verify_gdc_download.sh
```

To retrieve and rebuild the locked PanCancer Atlas labels before model fitting:

```bash
bash scripts/fetch_pancanatlas_pam50.sh
python3 scripts/lock_pancanatlas_pam50.py
```

The downloader skips files whose MD5 already matches the manifest, resumes
partial downloads, retries transient errors, and promotes a `.part` file only
after both byte-size and MD5 checks succeed.

## Expression matrices

Case-level matrices are stored in `processed/expression/`. They were built from
GDC STAR Counts after selecting one Primary Tumor file per TCGA case.

Matrix dimensions and types:

| Artifact | Shape | dtype | Meaning |
|---|---:|---|---|
| `counts_uint32.npy` | 1,095 × 60,660 | uint32 | GDC `unstranded` raw gene counts |
| `tpm_float32.npy` | 1,095 × 60,660 | float32 | GDC `tpm_unstranded` |
| `log2_tpm_float32.npy` | 1,095 × 60,660 | float32 | `log2(TPM + 1)` |

Axis and audit files:

- `samples.tsv`: matrix row metadata, labels, QC values, and annotation flags;
- `genes.tsv`: matrix column metadata and Ensembl identifiers;
- `candidate_file_audit.tsv`: selection audit for all 1,111 input files;
- `protein_coding_gene_indices.npy`: indices of 19,962 protein-coding genes;
- `pam50_labelled_sample_indices.npy`: indices of 981 labelled cases;
- `pam50_four_class_sample_indices.npy`: indices of 945 primary-analysis cases;
- `pam50_four_class_eligible_sample_indices.npy`: four-class indices after
  critical GDC annotation exclusions;
- `matrix_manifest.json`: versions, selection rules, dimensions, and SHA-256
  hashes for every matrix artifact;
- `verification_report.json`: independent matrix verification result.

No low-expression or variance filtering was applied while building these
matrices. Those data-dependent transformations must be fitted within each
training fold during model evaluation. The precomputed log transform is
sample-wise and does not learn parameters from other samples.

One case, `TCGA-A7-A0DC`, is retained for audit but marked
`analysis_eligible=0` because the GDC notes that the tumour is DCIS rather than
the invasive disease required by the TCGA-BRCA study. It has no locked PAM50
label, so the four-class count remains 945.

To rebuild and verify the matrices:

```bash
python3 scripts/build_expression_matrix.py
python3 scripts/verify_expression_matrix.py
```

The scripts require the versions recorded in `requirements-expression.txt`.
The `.npy` matrices are ignored by Git because their combined size is about
765 MiB; their metadata, manifest, audit table, and verification report remain
trackable.

Example memory-mapped loading:

```python
import numpy as np
import pandas as pd

root = "data/processed/expression"
X = np.load(f"{root}/log2_tpm_float32.npy", mmap_mode="r")
sample_indices = np.load(f"{root}/pam50_four_class_eligible_sample_indices.npy")
gene_indices = np.load(f"{root}/protein_coding_gene_indices.npy")
samples = pd.read_csv(f"{root}/samples.tsv", sep="\t")
genes = pd.read_csv(f"{root}/genes.tsv", sep="\t")

y = samples.loc[sample_indices, "pam50_standard"].to_numpy()
```

## Source endpoints

- <https://api.gdc.cancer.gov/status>
- <https://api.gdc.cancer.gov/files>
- <https://api.gdc.cancer.gov/cases>
- <https://api.gdc.cancer.gov/annotations>
- <https://api.gdc.cancer.gov/data/>
- <https://gdc.cancer.gov/about-data/publications/brca_2012>
