#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$CODE_ROOT}"
TASK="${TASK:-adjust_bottle}"
TASK_LIST="${TASK_LIST:-${TASK}}"
MODE="${MODE:-easy}"
EPISODES="${EPISODES:-50}"
START_EPISODE="${START_EPISODE:-0}"
SAVE_EVERY="${SAVE_EVERY:-5}"
ORIGINAL_IMAGE_KEY="${ORIGINAL_IMAGE_KEY:-rgb}"
POLICY_IMAGE_KEY="${POLICY_IMAGE_KEY:-default_isp}"
RUN_TAG="${RUN_TAG:-policy_easy_${TASK}_$(date -u +%Y%m%dT%H%M%SZ)}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${CODE_ROOT}/third_party/RoboTwin}"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${DATA_ROOT}/benchmark_data/robotwin2/dataset}"
ROBOTWIN_CHECKPOINT="${ROBOTWIN_CHECKPOINT:-${DATA_ROOT}/starvla_robotwin2_checkpoints/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-$(command -v python)}"
STARVLA_DIR="${STARVLA_DIR:-${CODE_ROOT}/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-$(command -v python)}"
POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-127.0.0.1}"
POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-5694}"
POLICY_AUDIT_H5PY_SITE="${POLICY_AUDIT_H5PY_SITE:-}"
EVAL_START_SEED="${EVAL_START_SEED:-}"
FORCE_EVAL_SEED="${FORCE_EVAL_SEED:-0}"
MATCH_DATASET_SEEDS="${MATCH_DATASET_SEEDS:-0}"
RECORD_EPISODE_DATA="${RECORD_EPISODE_DATA:-1}"
USE_DATASET_INSTRUCTIONS="${USE_DATASET_INSTRUCTIONS:-${MATCH_DATASET_SEEDS}}"
RESULT_DIR="${RESULT_DIR:-${DATA_ROOT}/results/robotwin2-policy-closed-loop/${RUN_TAG}}"
POLICY_TRAJECTORY_ROOT="${POLICY_TRAJECTORY_ROOT:-${RESULT_DIR}/trajectory_data}"

[[ "${MODE}" == easy ]] || { echo "Only MODE=easy is allowed by this runner" >&2; exit 2; }
[[ "${ORIGINAL_IMAGE_KEY}" == rgb ]] || { echo "Easy baseline requires ORIGINAL_IMAGE_KEY=rgb" >&2; exit 2; }
[[ "${POLICY_IMAGE_KEY}" == default_isp ]] || { echo "RAW replay requires POLICY_IMAGE_KEY=default_isp (rgb is raw_view8)" >&2; exit 2; }
[[ ! -e "${RESULT_DIR}" ]] || { echo "Refusing to overwrite existing result: ${RESULT_DIR}" >&2; exit 3; }
[[ -x "${ROBOTWIN_PYTHON}" ]] || { echo "Missing RoboTwin Python: ${ROBOTWIN_PYTHON}" >&2; exit 4; }
[[ -x "${STARVLA_PYTHON}" ]] || { echo "Missing StarVLA Python: ${STARVLA_PYTHON}" >&2; exit 4; }
[[ -f "${ROBOTWIN_CHECKPOINT}" ]] || { echo "Missing checkpoint: ${ROBOTWIN_CHECKPOINT}" >&2; exit 4; }
if [[ -n "${POLICY_AUDIT_H5PY_SITE}" ]]; then
  [[ -f "${POLICY_AUDIT_H5PY_SITE}/h5py/__init__.py" ]] || { echo "Missing compatible audit h5py: ${POLICY_AUDIT_H5PY_SITE}" >&2; exit 4; }
fi

IFS=',' read -ra tasks <<< "${TASK_LIST}"
for task in "${tasks[@]}"; do
  data_dir="${ROBOTWIN_DATA_ROOT}/${task}/aloha-agilex_clean_50"
  [[ -d "${data_dir}/data" && -f "${data_dir}/seed.txt" ]] || { echo "Missing clean_50 data: ${data_dir}" >&2; exit 4; }
  [[ -f "${ROBOTWIN_ROOT}/envs/${task}.py" ]] || { echo "Missing task source: ${task}" >&2; exit 4; }
done

EVAL_SEEDS=""
if [[ "${MATCH_DATASET_SEEDS}" == "1" ]]; then
  [[ "${#tasks[@]}" -eq 1 ]] || { echo "MATCH_DATASET_SEEDS=1 requires one task per run" >&2; exit 2; }
  EVAL_SEEDS="$(${ROBOTWIN_PYTHON} - "${ROBOTWIN_DATA_ROOT}/${tasks[0]}/aloha-agilex_clean_50/seed.txt" "${START_EPISODE}" "${EPISODES}" <<'PY'
import sys
values=[int(x) for x in open(sys.argv[1]).read().split()]
start,count=int(sys.argv[2]),int(sys.argv[3])
selected=values[start:start+count]
if len(selected)!=count: raise SystemExit(f"Requested {count} seeds at {start}, found {len(selected)}")
print(','.join(map(str,selected)))
PY
)"
fi

mkdir -p "${RESULT_DIR}/audit"
"${ROBOTWIN_PYTHON}" "${CODE_ROOT}/scripts/write_robotwin2_policy_metadata.py" \
  --output "${RESULT_DIR}/run_metadata.json" --run-tag "${RUN_TAG}" --tasks "${TASK_LIST}" \
  --episodes "${EPISODES}" --start-episode "${START_EPISODE}" --save-every "${SAVE_EVERY}" \
  --eval-start-seed "${EVAL_START_SEED}" \
  --eval-seeds "${EVAL_SEEDS}" --match-dataset-seeds "${MATCH_DATASET_SEEDS}" \
  --original-image-key "${ORIGINAL_IMAGE_KEY}" --policy-image-key "${POLICY_IMAGE_KEY}" \
  --checkpoint "${ROBOTWIN_CHECKPOINT}" --result-dir "${RESULT_DIR}"

export CODE_ROOT STARVLA_DIR STARVLA_PYTHON ROBOTWIN_PYTHON
NINJA_BIN_DIR="${NINJA_BIN_DIR:-$(dirname "${ROBOTWIN_PYTHON}")}"
export PATH="$(dirname "${ROBOTWIN_PYTHON}"):${NINJA_BIN_DIR}:${PATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/rawvla_torch_extensions_robotwin_policy_local}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
export PYTHONPATH="${ROBOTWIN_ROOT}/envs/curobo/src:${STARVLA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export ROBOTWIN_PATH="${ROBOTWIN_ROOT}" RESULT_ROOT="${RESULT_DIR}/engine"
export CKPTS_CSV=Qwen3-VL-OFT-RoboTwin2-All MODES_CSV=easy TASKS_CSV="${TASK_LIST}"
export SMOKE_TRIALS="${EPISODES}" START_EPISODE
export BASE_PORT="${POLICY_SERVER_PORT}" POLICY_SERVER_HOST
export ROBOTWIN_IMAGE_OBS_KEY="${POLICY_IMAGE_KEY}"
export ROBOTWIN_POLICY_AUDIT_DIR="${RESULT_DIR}/audit"
export ROBOTWIN_POLICY_AUDIT_DATA_ROOT="${ROBOTWIN_DATA_ROOT}"
export ROBOTWIN_USE_DATASET_INSTRUCTIONS="${USE_DATASET_INSTRUCTIONS}"
export ROBOTWIN_POLICY_RECORD_EPISODE_DATA="${RECORD_EPISODE_DATA}"
export ROBOTWIN_POLICY_TRAJECTORY_ROOT="${POLICY_TRAJECTORY_ROOT}"
export ROBOTWIN_POLICY_AUDIT_ORIGINAL_KEY="${ORIGINAL_IMAGE_KEY}"
export ROBOTWIN_POLICY_AUDIT_POLICY_KEY="${POLICY_IMAGE_KEY}"
export ROBOTWIN_POLICY_CHECKPOINT="${ROBOTWIN_CHECKPOINT}"
export ROBOTWIN_POLICY_AUDIT_SAVE_EVERY="${SAVE_EVERY}"
export ROBOTWIN_POLICY_AUDIT_H5PY_SITE="${POLICY_AUDIT_H5PY_SITE}"
if [[ -n "${EVAL_START_SEED}" ]]; then
  export ROBOTWIN_EVAL_START_SEED="${EVAL_START_SEED}"
fi
export ROBOTWIN_FORCE_EVAL_SEED="${FORCE_EVAL_SEED}"
if [[ -n "${EVAL_SEEDS}" ]]; then
  export ROBOTWIN_EVAL_SEEDS="${EVAL_SEEDS}"
fi
export RAWVLA_QWEN_ENABLE_FAST_HIDDEN=0
export STARVLA_REQUIRE_FLOAT_IMAGE=1

bash "${CODE_ROOT}/scripts/run_robotwin2_starvla_rgb_eval.sh" smoke
"${ROBOTWIN_PYTHON}" "${CODE_ROOT}/scripts/summarize_robotwin2_policy_rollout.py" "${RESULT_DIR}"
