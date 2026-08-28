#!/usr/bin/env bash
# Fetch the GridSet benchmark data from Zenodo into this directory.
#
# Record: https://doi.org/10.5281/zenodo.14451473  (CC-BY-4.0)
# The token in the upstream Readme link is not required -- the record is public.
#
# Usage:  bash data/fetch_gridset.sh [--no-pred-only]

set -euo pipefail

REC=14451473
BASE="https://zenodo.org/api/records/${REC}/files"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# file:expected_md5:expected_bytes
FILES=(
  "GridSet_no_pred.csv:e39601ea97573a406eb788c9940943e2:8629854"
  "GridSet_with_pred.csv:16c70f9f5a39c1342013947ddee436f1:100742406"
)

[[ "${1:-}" == "--no-pred-only" ]] && FILES=("${FILES[0]}")

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}'
  else md5 -q "$1"; fi
}

for entry in "${FILES[@]}"; do
  IFS=':' read -r name want_md5 want_size <<< "$entry"
  dest="${DIR}/${name}"

  if [[ -f "$dest" ]] && [[ "$(md5_of "$dest")" == "$want_md5" ]]; then
    echo "ok (cached)   ${name}"
    continue
  fi

  echo "downloading   ${name} (${want_size} bytes)..."
  curl -fL --progress-bar -o "$dest" "${BASE}/${name}/content"

  got_md5="$(md5_of "$dest")"
  got_size="$(wc -c < "$dest" | tr -d ' ')"
  if [[ "$got_md5" != "$want_md5" || "$got_size" != "$want_size" ]]; then
    echo "FAILED checksum for ${name}" >&2
    echo "  expected md5=${want_md5} size=${want_size}" >&2
    echo "  got      md5=${got_md5} size=${got_size}" >&2
    exit 1
  fi
  echo "ok (verified) ${name}"
done

echo
echo "GridSet ready in ${DIR}"
