#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${project_root}/data/manifests/gdc_manifest_tcga_brca_star_counts.tsv"
download_root="${project_root}/data/raw/gdc/star_counts"
log_dir="${project_root}/logs"
validator="${project_root}/scripts/verify_gdc_download.py"
python_bin="${PYTHON:-python3}"

mkdir -p "${download_root}" "${log_dir}"

for required_command in bash curl xargs "${python_bin}"; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "Required command not found: ${required_command}" >&2
    exit 127
  fi
done

download_workers="${GDC_DOWNLOAD_WORKERS:-12}"
if [[ ! "${download_workers}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GDC_DOWNLOAD_WORKERS must be a positive integer: ${download_workers}" >&2
  exit 2
fi

# Validate schema, UUIDs, digests, sizes, states, and safe path components
# before using any manifest value to construct a local path.
"${python_bin}" "${validator}" manifest --manifest "${manifest}" --quiet

verify_file() {
  "${python_bin}" "${validator}" file \
    --path "$1" \
    --expected-md5 "$2" \
    --expected-size "$3" \
    --quiet
}

download_one() {
  local file_id="$1"
  local file_name="$2"
  local expected_md5="$3"
  local expected_size="$4"
  local target_dir="${download_root}/${file_id}"
  local target="${target_dir}/${file_name}"
  local partial="${target}.part"

  mkdir -p "${target_dir}"

  if [[ -f "${target}" ]] && verify_file \
    "${target}" "${expected_md5}" "${expected_size}"; then
    printf 'SKIP\t%s\t%s\n' "${file_id}" "${file_name}"
    return 0
  fi

  if [[ -f "${partial}" ]] && verify_file \
    "${partial}" "${expected_md5}" "${expected_size}"; then
    mv "${partial}" "${target}"
    printf 'OK\t%s\t%s\n' "${file_id}" "${file_name}"
    return 0
  fi

  # --retry-connrefused is available on older Linux curl releases where
  # --retry-all-errors is not, while --retry still covers transient HTTP errors.
  curl --fail --location --show-error \
    --silent \
    --retry 8 --retry-connrefused --retry-delay 5 \
    --continue-at - \
    --output "${partial}" \
    "https://api.gdc.cancer.gov/data/${file_id}"

  if ! verify_file "${partial}" "${expected_md5}" "${expected_size}"; then
    printf 'Integrity validation failed for %s (%s)\n' \
      "${file_id}" "${file_name}" >&2
    return 1
  fi

  mv "${partial}" "${target}"
  printf 'OK\t%s\t%s\n' "${file_id}" "${file_name}"
}

export project_root manifest download_root log_dir validator python_bin
export -f verify_file download_one

while IFS=$'\t' read -r file_id file_name expected_md5 expected_size state; do
  printf '%s\0%s\0%s\0%s\0' \
    "${file_id}" "${file_name}" "${expected_md5}" "${expected_size}"
done < <(tail -n +2 "${manifest}") \
  | xargs -0 -P "${download_workers}" -n 4 \
    bash -c 'download_one "$1" "$2" "$3" "$4"' _ \
  > "${log_dir}/gdc_star_counts_download.log" \
  2> "${log_dir}/gdc_star_counts_download.err"

echo "Download completed. Verifying manifest completeness, sizes, and MD5 digests..."
"${python_bin}" "${validator}" verify \
  --manifest "${manifest}" \
  --download-root "${download_root}" \
  --no-write-reports
