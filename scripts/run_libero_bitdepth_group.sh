#!/usr/bin/env bash
set -euo pipefail

GROUP_FILE="${1:?usage: run_libero_bitdepth_group.sh SETTINGS_TSV}"
BASE_PORT="${BASE_PORT:-18000}"
GPU="${GPU:-0}"
export GPU

idx=0
while IFS=$'\t' read -r model suite bits; do
  [[ -z "${model:-}" || "$model" == model ]] && continue
  export PORT=$((BASE_PORT + idx))
  echo "[$(date -Is)] group item start idx=$idx model=$model suite=$suite bits=$bits gpu=$GPU port=$PORT"
  attempts="${RAWVLA_GROUP_RETRIES:-1}"
  attempt=1
  while :; do
    set +e
    bash "$(dirname "$0")/run_libero_bitdepth_setting.sh" "$model" "$suite" "$bits"
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
      break
    fi
    if [[ "$attempt" -ge "$attempts" ]]; then
      echo "[$(date -Is)] group item failed idx=$idx model=$model suite=$suite bits=$bits attempt=$attempt/$attempts rc=$rc"
      exit "$rc"
    fi
    echo "[$(date -Is)] group item retry idx=$idx model=$model suite=$suite bits=$bits attempt=$attempt/$attempts rc=$rc"
    sleep "${RAWVLA_GROUP_RETRY_SLEEP:-10}"
    attempt=$((attempt + 1))
  done
  echo "[$(date -Is)] group item done idx=$idx model=$model suite=$suite bits=$bits"
  idx=$((idx + 1))
done <"$GROUP_FILE"
