#!/usr/bin/env bash
set -euo pipefail

BASE="/mnt/d/skin_cancer_project/datasets"
PROJECT="/home/byalc/phase1_project"
BCC_LIST="$PROJECT/data/manifests/cobra_bcc_additional_129.txt"
SCC_LIST="$PROJECT/data/manifests/cobra_ood_scc_missing_19.txt"
BCC_DIR="$BASE/cobra_bcc"
OOD_DIR="$BASE/cobra_ood/images"
LOG_DIR="$PROJECT/results"
BCC_JOBS="${BCC_JOBS:-12}"
SCC_JOBS="${SCC_JOBS:-8}"
mkdir -p "$BCC_DIR" "$OOD_DIR" "$LOG_DIR"

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found"
  exit 1
fi

if ! command -v xargs >/dev/null 2>&1; then
  echo "xargs not found"
  exit 1
fi

echo "Starting COBRA top-up download"
echo "BCC list: $BCC_LIST"
echo "SCC list: $SCC_LIST"
echo "BCC parallel jobs: $BCC_JOBS"
echo "SCC parallel jobs: $SCC_JOBS"

count_targets() {
  local list_path="$1"
  if [[ ! -f "$list_path" ]]; then
    echo 0
    return
  fi
  grep -cve '^$' "$list_path"
}

download_parallel() {
  local list_path="$1"
  local s3_prefix="$2"
  local out_dir="$3"
  local label="$4"
  local jobs="$5"
  local total
  total=$(count_targets "$list_path")

  echo "[$label] target files: $total"
  if [[ "$total" -eq 0 ]]; then
    return
  fi

  export AWS_PAGER=""
  export S3_PREFIX="$s3_prefix"
  export OUT_DIR="$out_dir"
  export LABEL="$label"

  < "$list_path" tr -d '\r' | grep -v '^$' | xargs -I{} -P "$jobs" bash -c '
    f="$1"
    out="$OUT_DIR/$f"
    if [[ -f "$out" ]]; then
      echo "[$LABEL] skip $f"
      exit 0
    fi
    echo "[$LABEL] download $f"
    aws s3 cp --no-sign-request "$S3_PREFIX/$f" "$out"
  ' _ {}
}

download_parallel "$BCC_LIST" "s3://cobra-pathology/packages/bcc/images" "$BCC_DIR" "BCC" "$BCC_JOBS"
download_parallel "$SCC_LIST" "s3://cobra-pathology/packages/ood/images" "$OOD_DIR" "SCC" "$SCC_JOBS"

echo "Done"
echo "BCC count: $(find "$BCC_DIR" -maxdepth 1 -name '*.tif' | wc -l)"
echo "OOD count: $(find "$OOD_DIR" -maxdepth 1 -name '*.tif' | wc -l)"