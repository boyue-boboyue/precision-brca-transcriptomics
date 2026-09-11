#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${project_root}/data/manifests/gdc_manifest_tcga_brca_star_counts.tsv"
download_root="${project_root}/data/raw/gdc/star_counts"
log_dir="${project_root}/logs"

mkdir -p "${download_root}" "${log_dir}"

if [[ ! -s "${manifest}" ]]; then
  echo "Manifest is missing or empty: ${manifest}" >&2
  exit 1
fi

download_one() {
  local file_id="$1"
  local file_name="$2"
  local expected_md5="$3"
  local expected_size="$4"
  local target_dir="${download_root}/${file_id}"
  local target="${target_dir}/${file_name}"
  local partial="${target}.part"

  mkdir -p "${target_dir}"

  if [[ -f "${target}" ]]; then
    local actual_md5
    actual_md5="$(md5 -q "${target}")"
    if [[ "${actual_md5}" == "${expected_md5}" ]]; then
      printf 'SKIP\t%s\t%s\n' "${file_id}" "${file_name}"
      return 0
    fi
  fi

  curl --fail --location --show-error \
    --silent \
    --retry 8 --retry-all-errors --retry-delay 5 \
    --continue-at - \
    --output "${partial}" \
    "https://api.gdc.cancer.gov/data/${file_id}"

  local actual_size actual_md5
  actual_size="$(stat -f '%z' "${partial}")"
  actual_md5="$(md5 -q "${partial}")"

  if [[ "${actual_size}" != "${expected_size}" ]]; then
    printf 'Size mismatch for %s: expected %s, got %s\n' \
      "${file_id}" "${expected_size}" "${actual_size}" >&2
    return 1
  fi
  if [[ "${actual_md5}" != "${expected_md5}" ]]; then
    printf 'MD5 mismatch for %s: expected %s, got %s\n' \
      "${file_id}" "${expected_md5}" "${actual_md5}" >&2
    return 1
  fi

  mv "${partial}" "${target}"
  printf 'OK\t%s\t%s\n' "${file_id}" "${file_name}"
}

export project_root manifest download_root log_dir
export -f download_one

download_workers="${GDC_DOWNLOAD_WORKERS:-12}"

tail -n +2 "${manifest}" \
  | xargs -P "${download_workers}" -n 5 bash -c 'download_one "$0" "$1" "$2" "$3"' \
  > "${log_dir}/gdc_star_counts_download.log" \
  2> "${log_dir}/gdc_star_counts_download.err"

echo "Download completed. Verifying file count..."
expected_count="$(tail -n +2 "${manifest}" | wc -l | tr -d ' ')"
actual_count="$(find "${download_root}" -type f ! -name '*.part' | wc -l | tr -d ' ')"

if [[ "${expected_count}" != "${actual_count}" ]]; then
  echo "File count mismatch: expected ${expected_count}, got ${actual_count}" >&2
  exit 1
fi

echo "Verified ${actual_count} files."
