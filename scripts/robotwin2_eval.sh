#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RAWVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${ROOT}/starVLA}"
ROBOTWIN_CHECKPOINT="${ROBOTWIN_CHECKPOINT:-}"

if [[ -z "${ROBOTWIN_CHECKPOINT}" ]]; then
  echo "Set ROBOTWIN_CHECKPOINT to the StarVLA RoboTwin checkpoint." >&2
  exit 2
fi
if [[ ! -d "${STARVLA_DIR}" ]]; then
  echo "STARVLA_DIR does not exist: ${STARVLA_DIR}" >&2
  exit 2
fi

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
