#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
study_id="brca_tcga_pan_can_atlas_2018"
api_root="https://www.cbioportal.org/api"
raw_dir="${project_root}/data/raw/pancanatlas/cbioportal/${study_id}"
stage_dir="$(mktemp -d)"
page_size=200

cleanup() {
  rm -rf "${stage_dir}"
}
trap cleanup EXIT

mkdir -p "${raw_dir}/subtype_pages"

curl --fail --silent --show-error --location \
  --retry 8 --retry-all-errors --retry-delay 3 \
  --output "${stage_dir}/study.json" \
  "${api_root}/studies/${study_id}"

curl --fail --silent --show-error --location \
  --retry 8 --retry-all-errors --retry-delay 3 \
  --output "${stage_dir}/clinical_attributes.json" \
  "${api_root}/studies/${study_id}/clinical-attributes"

jq -e --arg study_id "${study_id}" \
  '.studyId == $study_id' "${stage_dir}/study.json" >/dev/null
jq -e --arg study_id "${study_id}" '
  any(.[];
    .studyId == $study_id
    and .clinicalAttributeId == "SUBTYPE"
    and .patientAttribute == true
  )
' "${stage_dir}/clinical_attributes.json" >/dev/null

page=0
page_count=0
record_count=0
while :; do
  page_file="${stage_dir}/page_$(printf '%03d' "${page}").json"
  endpoint="${api_root}/studies/${study_id}/clinical-data?clinicalDataType=PATIENT&attributeId=SUBTYPE&projection=DETAILED&pageSize=${page_size}&pageNumber=${page}"

  curl --fail --silent --show-error --location \
    --retry 8 --retry-all-errors --retry-delay 3 \
    --output "${page_file}" \
    "${endpoint}"

  jq -e 'type == "array"' "${page_file}" >/dev/null
  current_count="$(jq 'length' "${page_file}")"
  if [[ "${current_count}" -eq 0 ]]; then
    rm "${page_file}"
    break
  fi

  page_count=$((page_count + 1))
  record_count=$((record_count + current_count))
  if [[ "${current_count}" -lt "${page_size}" ]]; then
    break
  fi
  page=$((page + 1))
done

jq -s 'add' "${stage_dir}"/page_*.json \
  > "${stage_dir}/pam50_subtype_records.json"

jq -e '
  length > 0
  and all(.[ ];
    .clinicalAttributeId == "SUBTYPE"
    and .patientAttribute == true
    and (.patientId | type == "string")
    and (.value | type == "string")
  )
' "${stage_dir}/pam50_subtype_records.json" >/dev/null

merged_count="$(jq 'length' "${stage_dir}/pam50_subtype_records.json")"
if [[ "${merged_count}" -ne "${record_count}" ]]; then
  echo "Merged record count mismatch: ${merged_count} != ${record_count}" >&2
  exit 1
fi

retrieved_at_utc="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
jq -n \
  --arg retrieved_at_utc "${retrieved_at_utc}" \
  --arg study_id "${study_id}" \
  --arg api_root "${api_root}" \
  --argjson page_size "${page_size}" \
  --argjson page_count "${page_count}" \
  --argjson record_count "${record_count}" \
  '{
    retrieved_at_utc: $retrieved_at_utc,
    provider: "cBioPortal",
    study_id: $study_id,
    study_page_url: ("https://www.cbioportal.org/study/summary?id=" + $study_id),
    original_data_url: "https://gdc.cancer.gov/about-data/publications/pancanatlas",
    api_root: $api_root,
    source_api_endpoint: ($api_root + "/studies/" + $study_id + "/clinical-data?clinicalDataType=PATIENT&attributeId=SUBTYPE&projection=DETAILED"),
    clinical_data_type: "PATIENT",
    attribute_id: "SUBTYPE",
    page_size: $page_size,
    page_count: $page_count,
    record_count: $record_count
  }' > "${stage_dir}/source_metadata.json"

cp "${stage_dir}/study.json" "${raw_dir}/study.json"
cp "${stage_dir}/clinical_attributes.json" "${raw_dir}/clinical_attributes.json"
cp "${stage_dir}/pam50_subtype_records.json" "${raw_dir}/pam50_subtype_records.json"
cp "${stage_dir}/source_metadata.json" "${raw_dir}/source_metadata.json"
cp "${stage_dir}"/page_*.json "${raw_dir}/subtype_pages/"

jq . "${raw_dir}/source_metadata.json"
jq -r 'group_by(.value)[] | [.[0].value, length] | @tsv' \
  "${raw_dir}/pam50_subtype_records.json"
