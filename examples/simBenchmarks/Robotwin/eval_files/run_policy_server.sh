#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash examples/simBenchmarks/Robotwin/eval_files/run_policy_server.sh <ckpt_path> [gpu_id] [port]" >&2
    exit 1
fi

your_ckpt="$1"
gpu_id="${2:-${ROBOTWIN_SERVER_GPU:-0}}"
port="${3:-${ROBOTWIN_SERVER_PORT:-5694}}"
star_vla_python="${STARVLA_PYTHON:-${star_vla_python:-python}}"

use_bf16_flag=()
if [[ "${ROBOTWIN_USE_BF16:-1}" != "0" ]]; then
    use_bf16_flag+=(--use_bf16)
fi

config_override_flags=()
ckpt_root="$(cd "$(dirname "${your_ckpt}")/.." && pwd)"
ckpt_config="${ckpt_root}/config.yaml"
if [[ -f "${ckpt_config}" ]]; then
    configured_base_vlm="$(awk '/^[[:space:]]*base_vlm:/ {sub(/^[^:]*:[[:space:]]*/, ""); print; exit}' "${ckpt_config}")"
    configured_base_vlm="${configured_base_vlm%\"}"
    configured_base_vlm="${configured_base_vlm#\"}"
    if [[ -n "${configured_base_vlm}" ]]; then
        if [[ "${configured_base_vlm}" = /* ]]; then
            resolved_base_vlm="${configured_base_vlm}"
        else
            resolved_base_vlm="${REPO_ROOT}/${configured_base_vlm}"
        fi
        local_base_vlm="${REPO_ROOT}/../checkpoints/base/${configured_base_vlm##*/}"
        if [[ ! -e "${resolved_base_vlm}" && -d "${local_base_vlm}" ]]; then
            echo "[INFO] overriding missing base_vlm with ${local_base_vlm}"
            config_override_flags+=(--config_override "framework.qwenvl.base_vlm=${local_base_vlm}")
        fi
    fi
fi

echo "[INFO] Starting RoboTwin policy server"
echo "[INFO] checkpoint: ${your_ckpt}"
echo "[INFO] gpu: ${gpu_id}"
echo "[INFO] port: ${port}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${star_vla_python}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    "${config_override_flags[@]}" \
    "${use_bf16_flag[@]}"
