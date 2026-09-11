#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
study_id="brca_tcga"
api_root="https://www.cbioportal.org/api"
raw_dir="${project_root}/data/raw/cbioportal/${study_id}/receptors"
stage_dir="$(mktemp -d)"
page_size=500
attributes=(ER_STATUS_BY_IHC PR_STATUS_BY_IHC IHC_HER2)

cleanup() {
  rm -rf "${stage_dir}"
}
trap cleanup EXIT

mkdir -p "${raw_dir}"

curl --fail --silent --show-error --location \
  --retry 8 --retry-all-errors --retry-delay 2 \
  --output "${stage_dir}/study.json" \
  "${api_root}/studies/${study_id}"

curl --fail --silent --show-error --location \
  --retry 8 --retry-all-errors --retry-delay 2 \
  --output "${stage_dir}/clinical_attributes.json" \
  "${api_root}/studies/${study_id}/clinical-attributes"

jq -e --arg study_id "${study_id}" '.studyId == $study_id' \
  "${stage_dir}/study.json" >/dev/null

for attribute_id in "${attributes[@]}"; do
  jq -e --arg study_id "${study_id}" --arg attribute_id "${attribute_id}" '
    any(.[ ];
      .studyId == $study_id
      and .clinicalAttributeId == $attribute_id
      and .patientAttribute == true
    )
  ' "${stage_dir}/clinical_attributes.json" >/dev/null

  page_dir="${stage_dir}/${attribute_id}_pages"
  mkdir -p "${page_dir}"
  page=0
  while :; do
    page_file="${page_dir}/page_$(printf '%03d' "${page}").json"
    endpoint="${api_root}/studies/${study_id}/clinical-data?clinicalDataType=PATIENT&attributeId=${attribute_id}&projection=DETAILED&pageSize=${page_size}&pageNumber=${page}"
    curl --fail --silent --show-error --location \
      --retry 8 --retry-all-errors --retry-delay 2 \
      --output "${page_file}" "${endpoint}"
    jq -e 'type == "array"' "${page_file}" >/dev/null
    current_count="$(jq 'length' "${page_file}")"
    if [[ "${current_count}" -eq 0 ]]; then
      rm "${page_file}"
      break
    fi
    if [[ "${current_count}" -lt "${page_size}" ]]; then
      break
    fi
    page=$((page + 1))
  done

  jq -s 'add' "${page_dir}"/page_*.json > "${stage_dir}/${attribute_id}.json"
  jq -e --arg attribute_id "${attribute_id}" '
    length > 0
    and all(.[ ];
      .clinicalAttributeId == $attribute_id
      and .patientAttribute == true
      and (.patientId | type == "string")
      and (.value | type == "string")
    )
  ' "${stage_dir}/${attribute_id}.json" >/dev/null
done

retrieved_at_utc="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
er_count="$(jq 'length' "${stage_dir}/ER_STATUS_BY_IHC.json")"
pr_count="$(jq 'length' "${stage_dir}/PR_STATUS_BY_IHC.json")"
her2_count="$(jq 'length' "${stage_dir}/IHC_HER2.json")"

jq -n \
  --arg retrieved_at_utc "${retrieved_at_utc}" \
  --arg study_id "${study_id}" \
  --arg api_root "${api_root}" \
  --argjson page_size "${page_size}" \
  --argjson er_count "${er_count}" \
  --argjson pr_count "${pr_count}" \
  --argjson her2_count "${her2_count}" \
  '{
    retrieved_at_utc: $retrieved_at_utc,
    provider: "cBioPortal",
    study_id: $study_id,
    study_page_url: ("https://www.cbioportal.org/study/summary?id=" + $study_id),
    api_root: $api_root,
    clinical_data_type: "PATIENT",
    page_size: $page_size,
    attributes: {
      ER_STATUS_BY_IHC: {
        record_count: $er_count,
        endpoint: ($api_root + "/studies/" + $study_id + "/clinical-data?clinicalDataType=PATIENT&attributeId=ER_STATUS_BY_IHC&projection=DETAILED")
      },
      PR_STATUS_BY_IHC: {
        record_count: $pr_count,
        endpoint: ($api_root + "/studies/" + $study_id + "/clinical-data?clinicalDataType=PATIENT&attributeId=PR_STATUS_BY_IHC&projection=DETAILED")
      },
      IHC_HER2: {
        record_count: $her2_count,
        endpoint: ($api_root + "/studies/" + $study_id + "/clinical-data?clinicalDataType=PATIENT&attributeId=IHC_HER2&projection=DETAILED")
      }
    }
  }' > "${stage_dir}/source_metadata.json"

for file_name in study.json clinical_attributes.json ER_STATUS_BY_IHC.json PR_STATUS_BY_IHC.json IHC_HER2.json source_metadata.json; do
  cp "${stage_dir}/${file_name}" "${raw_dir}/${file_name}"
done

(
  cd "${raw_dir}"
  shasum -a 256 \
    study.json \
    clinical_attributes.json \
    ER_STATUS_BY_IHC.json \
    PR_STATUS_BY_IHC.json \
    IHC_HER2.json \
    source_metadata.json > source_files.sha256
)

jq . "${raw_dir}/source_metadata.json"
for attribute_id in "${attributes[@]}"; do
  jq -r --arg attribute_id "${attribute_id}" '
    group_by(.value)[]
    | [$attribute_id, .[0].value, length]
    | @tsv
  ' "${raw_dir}/${attribute_id}.json"
done
