#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest_dir="${project_root}/data/manifests"
metadata_dir="${project_root}/data/metadata"

mkdir -p "${manifest_dir}" "${metadata_dir}"

curl --fail --silent --show-error \
  "https://api.gdc.cancer.gov/status" \
  > "${metadata_dir}/gdc_status.json"

curl --fail --silent --show-error \
  --request POST \
  --header "Content-Type: application/json" \
  --data-binary "@${manifest_dir}/gdc_star_counts_query.json" \
  "https://api.gdc.cancer.gov/files" \
  > "${metadata_dir}/gdc_star_counts_files.json"

curl --fail --silent --show-error \
  --request POST \
  --header "Content-Type: application/json" \
  --data-binary "@${manifest_dir}/gdc_cases_query.json" \
  "https://api.gdc.cancer.gov/cases" \
  > "${metadata_dir}/gdc_cases.json"

curl --fail --silent --show-error \
  --request POST \
  --header "Content-Type: application/json" \
  --data-binary "@${manifest_dir}/gdc_annotations_query.json" \
  "https://api.gdc.cancer.gov/annotations" \
  > "${metadata_dir}/gdc_annotations.json"

jq -e '.data.hits | type == "array"' \
  "${metadata_dir}/gdc_star_counts_files.json" >/dev/null
jq -e '.data.hits | type == "array"' \
  "${metadata_dir}/gdc_cases.json" >/dev/null
jq -e '.data.hits | type == "array"' \
  "${metadata_dir}/gdc_annotations.json" >/dev/null

jq -r '
  ["id", "filename", "md5", "size", "state"],
  (.data.hits[] | [.file_id, .file_name, .md5sum, .file_size, .state])
  | @tsv
' "${metadata_dir}/gdc_star_counts_files.json" \
  > "${manifest_dir}/gdc_manifest_tcga_brca_star_counts.tsv"

jq -r '
  [
    "file_id", "file_name", "md5sum", "file_size", "access",
    "workflow_type", "case_id", "case_barcode", "sample_id",
    "sample_barcode", "sample_type", "tissue_type", "aliquot_id",
    "aliquot_barcode"
  ],
  (
    .data.hits[] as $file
    | $file.cases[] as $case
    | $case.samples[]
    | select(.sample_type == "Primary Tumor") as $sample
    | ($sample.portions[]?.analytes[]?.aliquots[]? // {}) as $aliquot
    | [
        $file.file_id,
        $file.file_name,
        $file.md5sum,
        $file.file_size,
        $file.access,
        $file.analysis.workflow_type,
        $case.case_id,
        $case.submitter_id,
        $sample.sample_id,
        $sample.submitter_id,
        $sample.sample_type,
        $sample.tissue_type,
        ($aliquot.aliquot_id // ""),
        ($aliquot.submitter_id // "")
      ]
  )
  | @tsv
' "${metadata_dir}/gdc_star_counts_files.json" \
  > "${metadata_dir}/gdc_star_counts_file_sample_map.tsv"

jq '
  {
    query_timestamp_utc: (now | todateiso8601),
    file_count: (.data.hits | length),
    api_total: .data.pagination.total,
    total_bytes: ([.data.hits[].file_size] | add // 0),
    total_gib: (([.data.hits[].file_size] | add // 0) / 1073741824),
    unique_cases: ([.data.hits[].cases[].case_id] | unique | length)
  }
' "${metadata_dir}/gdc_star_counts_files.json" \
  > "${metadata_dir}/gdc_star_counts_summary.json"

(
  cd "${project_root}"
  shasum -a 256 \
    data/manifests/gdc_star_counts_query.json \
    data/manifests/gdc_cases_query.json \
    data/manifests/gdc_annotations_query.json \
    data/manifests/gdc_manifest_tcga_brca_star_counts.tsv \
    data/metadata/gdc_status.json \
    data/metadata/gdc_star_counts_files.json \
    data/metadata/gdc_cases.json \
    data/metadata/gdc_annotations.json \
    data/metadata/gdc_star_counts_file_sample_map.tsv \
    > data/metadata/query_artifacts.sha256
)

jq . "${metadata_dir}/gdc_status.json"
jq . "${metadata_dir}/gdc_star_counts_summary.json"
