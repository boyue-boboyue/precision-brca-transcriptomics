#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${project_root}/data/manifests/gdc_manifest_tcga_brca_star_counts.tsv"
download_root="${project_root}/data/raw/gdc/star_counts"
report="${project_root}/data/metadata/gdc_download_verification.tsv"
summary="${project_root}/data/metadata/gdc_download_verification_summary.json"

file_md5() {
  if command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  elif command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  else
    echo "Neither md5 nor md5sum is available" >&2
    return 127
  fi
}

file_size() {
  if stat -f '%z' "$1" >/dev/null 2>&1; then
    stat -f '%z' "$1"
  else
    stat -c '%s' "$1"
  fi
}

printf 'file_id\tfile_name\texpected_size\tactual_size\texpected_md5\tactual_md5\tstatus\n' \
  > "${report}"

expected_count=0
verified_count=0
failed_count=0
expected_bytes=0
actual_bytes=0

while IFS=$'\t' read -r file_id file_name expected_md5 expected_size state; do
  target="${download_root}/${file_id}/${file_name}"
  expected_count=$((expected_count + 1))
  expected_bytes=$((expected_bytes + expected_size))

  if [[ ! -f "${target}" ]]; then
    printf '%s\t%s\t%s\t\t%s\t\tMISSING\n' \
      "${file_id}" "${file_name}" "${expected_size}" "${expected_md5}" \
      >> "${report}"
    failed_count=$((failed_count + 1))
    continue
  fi

  observed_size="$(file_size "${target}")"
  observed_md5="$(file_md5 "${target}")"
  actual_bytes=$((actual_bytes + observed_size))

  if [[ "${observed_size}" == "${expected_size}" && "${observed_md5}" == "${expected_md5}" ]]; then
    status="OK"
    verified_count=$((verified_count + 1))
  else
    status="FAILED"
    failed_count=$((failed_count + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${file_id}" "${file_name}" "${expected_size}" "${observed_size}" \
    "${expected_md5}" "${observed_md5}" "${status}" \
    >> "${report}"
done < <(tail -n +2 "${manifest}")

jq -n \
  --arg verified_at_utc "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --argjson expected_count "${expected_count}" \
  --argjson verified_count "${verified_count}" \
  --argjson failed_count "${failed_count}" \
  --argjson expected_bytes "${expected_bytes}" \
  --argjson actual_bytes "${actual_bytes}" \
  '{
    verified_at_utc: $verified_at_utc,
    expected_file_count: $expected_count,
    verified_file_count: $verified_count,
    failed_file_count: $failed_count,
    expected_bytes: $expected_bytes,
    actual_bytes: $actual_bytes,
    expected_gib: ($expected_bytes / 1073741824),
    actual_gib: ($actual_bytes / 1073741824),
    all_files_verified: (
      $failed_count == 0
      and $expected_count == $verified_count
      and $expected_bytes == $actual_bytes
    )
  }' > "${summary}"

jq . "${summary}"

if [[ "${failed_count}" -ne 0 || "${expected_count}" -ne "${verified_count}" ]]; then
  exit 1
fi
