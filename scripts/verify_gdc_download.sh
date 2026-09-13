#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${python_bin}" >&2
  exit 127
fi

# The Python standard-library implementation is identical on Linux and macOS;
# it does not depend on BSD/GNU stat flags, md5/md5sum, jq, or shasum.
exec "${python_bin}" "${project_root}/scripts/verify_gdc_download.py" verify \
  --manifest "${project_root}/data/manifests/gdc_manifest_tcga_brca_star_counts.tsv" \
  --download-root "${project_root}/data/raw/gdc/star_counts" \
  --report "${project_root}/data/metadata/gdc_download_verification.tsv" \
  --summary "${project_root}/data/metadata/gdc_download_verification_summary.json" \
  "$@"
