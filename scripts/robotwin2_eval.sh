#!/usr/bin/env bash
set -euo pipefail

ROOT="/tmp/scratch-space/RawVLA"
source "${ROOT}/.bashrc"

mode="${1:-demo_clean}"
shift || true
if [[ "$#" -eq 0 ]]; then
  set -- all
fi

exec bash "${STARVLA_DIR}/examples/simBenchmarks/Robotwin/eval_files/start_eval.sh" \
  --mode "${mode}" \
  --name "Qwen3-VL-OFT-RoboTwin2-All" \
  --ckpt "${ROBOTWIN_CHECKPOINT}" \
  "$@"
