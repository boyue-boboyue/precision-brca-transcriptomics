#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input="${project_root}/data/metadata/gdc_cases.json"
annotations_input="${project_root}/data/metadata/gdc_annotations.json"
output_dir="${project_root}/data/metadata"

if [[ ! -s "${input}" ]]; then
  echo "Missing GDC cases response: ${input}" >&2
  exit 1
fi

jq -r '
  [
    "case_id", "case_barcode", "primary_site", "disease_type",
    "sex_at_birth", "age_at_index", "race", "ethnicity", "vital_status",
    "days_to_death", "primary_diagnosis", "morphology",
    "ajcc_pathologic_stage", "ajcc_pathologic_t", "ajcc_pathologic_n",
    "ajcc_pathologic_m", "age_at_diagnosis", "year_of_diagnosis",
    "diagnosis_id"
  ],
  (
    .data.hits[]
    | . as $case
    | (
        [.diagnoses[]? | select(.diagnosis_is_primary_disease == true)][0]
        // [.diagnoses[]? | select(.classification_of_tumor == "primary")][0]
        // [.diagnoses[]? | select(.days_to_diagnosis == 0)][0]
        // .diagnoses[0]
        // {}
      ) as $diagnosis
    | [
        $case.case_id,
        $case.submitter_id,
        $case.primary_site,
        $case.disease_type,
        ($case.demographic.sex_at_birth // ""),
        ($case.demographic.age_at_index // ""),
        ($case.demographic.race // ""),
        ($case.demographic.ethnicity // ""),
        ($case.demographic.vital_status // ""),
        ($case.demographic.days_to_death // ""),
        ($diagnosis.primary_diagnosis // ""),
        ($diagnosis.morphology // ""),
        ($diagnosis.ajcc_pathologic_stage // ""),
        ($diagnosis.ajcc_pathologic_t // ""),
        ($diagnosis.ajcc_pathologic_n // ""),
        ($diagnosis.ajcc_pathologic_m // ""),
        ($diagnosis.age_at_diagnosis // ""),
        ($diagnosis.year_of_diagnosis // ""),
        ($diagnosis.diagnosis_id // "")
      ]
  )
  | @tsv
' "${input}" > "${output_dir}/gdc_cases_clinical.tsv"

jq -r '
  [
    "case_id", "case_barcode", "sample_id", "sample_barcode", "sample_type",
    "tissue_type", "tumor_descriptor", "specimen_type", "preservation_method",
    "days_to_collection", "initial_weight"
  ],
  (
    .data.hits[] as $case
    | $case.samples[]?
    | [
        $case.case_id,
        $case.submitter_id,
        .sample_id,
        .submitter_id,
        .sample_type,
        .tissue_type,
        (.tumor_descriptor // ""),
        (.specimen_type // ""),
        (.preservation_method // ""),
        (.days_to_collection // ""),
        (.initial_weight // "")
      ]
  )
  | @tsv
' "${input}" > "${output_dir}/gdc_samples.tsv"

jq -r '
  [
    "case_id", "case_barcode", "sample_id", "sample_barcode", "portion_id",
    "analyte_id", "analyte_barcode", "analyte_type", "rna_integrity_number",
    "aliquot_id", "aliquot_barcode", "source_center"
  ],
  (
    .data.hits[] as $case
    | $case.samples[]? as $sample
    | $sample.portions[]? as $portion
    | $portion.analytes[]? as $analyte
    | $analyte.aliquots[]?
    | [
        $case.case_id,
        $case.submitter_id,
        $sample.sample_id,
        $sample.submitter_id,
        $portion.portion_id,
        $analyte.analyte_id,
        $analyte.submitter_id,
        ($analyte.analyte_type // ""),
        ($analyte.rna_integrity_number // ""),
        .aliquot_id,
        .submitter_id,
        (.source_center // "")
      ]
  )
  | @tsv
' "${input}" > "${output_dir}/gdc_aliquots.tsv"

if [[ -s "${annotations_input}" ]]; then
  jq -r '
    [
      "annotation_id", "case_id", "case_barcode", "entity_id",
      "entity_barcode", "entity_type", "category", "classification",
      "status", "notes", "created_datetime", "updated_datetime"
    ],
    (
      .data.hits[]
      | [
          .annotation_id,
          (.case_id // ""),
          (.case_submitter_id // ""),
          (.entity_id // ""),
          (.entity_submitter_id // ""),
          (.entity_type // ""),
          (.category // ""),
          (.classification // ""),
          (.status // ""),
          (.notes // ""),
          (.created_datetime // ""),
          (.updated_datetime // "")
        ]
    )
    | @tsv
  ' "${annotations_input}" > "${output_dir}/gdc_annotations.tsv"
fi

(
  cd "${project_root}"
  shasum -a 256 \
    data/metadata/gdc_cases_clinical.tsv \
    data/metadata/gdc_samples.tsv \
    data/metadata/gdc_aliquots.tsv \
    data/metadata/gdc_annotations.tsv \
    > data/metadata/metadata_tables.sha256
)

wc -l \
  "${output_dir}/gdc_cases_clinical.tsv" \
  "${output_dir}/gdc_samples.tsv" \
  "${output_dir}/gdc_aliquots.tsv" \
  "${output_dir}/gdc_annotations.tsv"
