#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROXY_HTTP="${PROXY_HTTP:-}"
PROXY_ALL="${PROXY_ALL:-}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/benchmark_data}"
LIBERO_DIR="${LIBERO_DIR:-${DATA_ROOT}/libero/LIBERO-datasets}"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-${DATA_ROOT}/robotwin2}"
STATUS_FILE="${STATUS_FILE:-${DATA_ROOT}/download_replay_datasets.status}"
MAX_WORKERS="${MAX_WORKERS:-8}"
LIBERO_INCLUDE=(
  "libero_10/*.hdf5"
  "libero_goal/*.hdf5"
  "libero_object/*.hdf5"
  "libero_spatial/*.hdf5"
  "README.md"
  ".gitattributes"
)

if [[ -n "${PROXY_HTTP}" ]]; then
  export https_proxy="${PROXY_HTTP}" http_proxy="${PROXY_HTTP}"
  export HTTPS_PROXY="${PROXY_HTTP}" HTTP_PROXY="${PROXY_HTTP}"
fi
if [[ -n "${PROXY_ALL}" ]]; then
  export all_proxy="${PROXY_ALL}" ALL_PROXY="${PROXY_ALL}"
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

on_error() {
  local rc=$?
  printf "failed rc=%s\n" "${rc}" > "${STATUS_FILE}"
  exit "${rc}"
}
trap on_error ERR

mkdir -p "${LIBERO_DIR}" "${ROBOTWIN_DIR}" "$(dirname "${STATUS_FILE}")"

printf "running_libero\n" > "${STATUS_FILE}"
echo "[$(date -Is)] proxy=${PROXY_HTTP:-disabled}"
echo "[$(date -Is)] Download LIBERO HDF5 -> ${LIBERO_DIR}"
hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir "${LIBERO_DIR}" \
  --include "${LIBERO_INCLUDE[@]}" \
  --max-workers "${MAX_WORKERS}"
echo "[$(date -Is)] LIBERO download complete"

printf "running_robotwin2\n" > "${STATUS_FILE}"
echo "[$(date -Is)] Download RoboTwin2.0 aloha-agilex_clean_50.zip files -> ${ROBOTWIN_DIR}"
hf download TianxingChen/RoboTwin2.0 \
  --repo-type dataset \
  --local-dir "${ROBOTWIN_DIR}" \
  --include "dataset/*/aloha-agilex_clean_50.zip" \
  --max-workers "${MAX_WORKERS}"
echo "[$(date -Is)] RoboTwin2.0 download complete"

printf "done\n" > "${STATUS_FILE}"
echo "[$(date -Is)] all downloads complete"
