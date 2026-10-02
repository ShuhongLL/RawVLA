#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$CODE_ROOT}"
FASTWAM_DIR="${FASTWAM_DIR:-${CODE_ROOT}/third_party/FastWAM}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${FASTWAM_DIR}/third_party/RoboTwin}"
CKPT_DIR="${CKPT_DIR:-${DATA_ROOT}/fastwam_robotwin2_checkpoint}"
CKPT_PATH="${CKPT_PATH:-${CKPT_DIR}/robotwin_uncond_3cam_384.pt}"
STATS_PATH="${STATS_PATH:-${CKPT_DIR}/robotwin_uncond_3cam_384_dataset_stats.json}"
RESULT_ROOT="${RESULT_ROOT:-${DATA_ROOT}/results/robotwin2-fastwam}"

NUM_GPUS="${NUM_GPUS:-1}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
SMOKE_TASK="${SMOKE_TASK:-adjust_bottle}"
SMOKE_EPISODES="${SMOKE_EPISODES:-1}"
FULL_EPISODES="${FULL_EPISODES:-${TRIALS:-50}}"
MODES_CSV="${MODES_CSV:-easy,hard}"
RUN_MODE="${1:-smoke}"
shift || true

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/run_fastwam_robotwin2_eval.sh [smoke|full|one] [task]

Modes:
  smoke  Run one task with one episode per clean/random phase.
  full   Run all RoboTwin 2.0 tasks with FULL_EPISODES per selected mode.
  one    Run one provided task with EVAL_EPISODES per phase.

Useful env:
  NUM_GPUS=1 MAX_TASKS_PER_GPU=1
  SMOKE_TASK=adjust_bottle SMOKE_EPISODES=1 TASKS_CSV=adjust_bottle,open_laptop
  MODES_CSV=easy,hard  # easy=demo_clean_rgb_eval, hard=demo_randomized_no_light_rgb_eval
  TRIALS=50 FULL_EPISODES=50 EVAL_EPISODES=10
  CUDA_VISIBLE_DEVICES=0
  CKPT_DIR=/path/to/fastwam_robotwin2_checkpoint
  FASTWAM_PYTHON=/path/to/python
EOF
}

if [[ "${RUN_MODE}" == "-h" || "${RUN_MODE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "full" && "${RUN_MODE}" != "one" ]]; then
  echo "Unsupported mode: ${RUN_MODE}" >&2
  usage
  exit 1
fi

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing directory: ${path}" >&2
    exit 1
  fi
}

require_dir "${FASTWAM_DIR}"
require_dir "${ROBOTWIN_ROOT}"
require_file "${CKPT_PATH}"
require_file "${STATS_PATH}"
require_file "${ROBOTWIN_ROOT}/task_config/_eval_step_limit.yml"

FASTWAM_PYTHON="${FASTWAM_PYTHON:-$(command -v python)}"
if [[ -z "${FASTWAM_PYTHON}" || ! -x "${FASTWAM_PYTHON}" ]]; then
  echo "Missing executable python. Set FASTWAM_PYTHON=/path/to/python." >&2
  exit 1
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "${value}"
}

join_by_comma() {
  local out=""
  local item
  for item in "$@"; do
    out="${out:+${out},}${item}"
  done
  printf '%s\n' "${out}"
}

hydra_list() {
  printf '[%s]\n' "$(join_by_comma "$@")"
}

write_runtime_task_configs() {
  local task_config_dir="${ROBOTWIN_ROOT}/task_config"
  local target tmp
  require_file "${task_config_dir}/demo_clean.yml"
  require_file "${task_config_dir}/demo_randomized.yml"

  target="${task_config_dir}/demo_clean_rgb_eval.yml"
  tmp="$(mktemp "${target}.tmp.XXXXXX")"
  sed \
    -e 's/^eval_video_log:.*/eval_video_log: false/' \
    "${task_config_dir}/demo_clean.yml" > "${tmp}"
  if cmp -s "${tmp}" "${target}" 2>/dev/null; then
    rm -f "${tmp}"
  else
    mv -f "${tmp}" "${target}"
  fi

  target="${task_config_dir}/demo_randomized_no_light_rgb_eval.yml"
  tmp="$(mktemp "${target}.tmp.XXXXXX")"
  sed \
    -e 's/^[[:space:]]*random_light:.*/  random_light: false/' \
    -e 's/^[[:space:]]*crazy_random_light_rate:.*/  crazy_random_light_rate: 0/' \
    -e 's/^eval_video_log:.*/eval_video_log: false/' \
    "${task_config_dir}/demo_randomized.yml" > "${tmp}"
  if cmp -s "${tmp}" "${target}" 2>/dev/null; then
    rm -f "${tmp}"
  else
    mv -f "${tmp}" "${target}"
  fi
}

resolve_task_name_override() {
  local -a selected=()
  if [[ -n "${TASKS_CSV:-}" ]]; then
    IFS=',' read -ra selected <<< "${TASKS_CSV}"
    printf '%s\n' "EVALUATION.task_name=$(hydra_list "${selected[@]}")"
    return 0
  fi
  case "${RUN_MODE}" in
    smoke)
      printf '%s\n' "EVALUATION.task_name=$(hydra_list "${1:-${SMOKE_TASK}}")"
      ;;
    one)
      if [[ $# -lt 1 ]]; then
        echo "Mode 'one' requires at least one task name." >&2
        usage
        exit 1
      fi
      printf '%s\n' "EVALUATION.task_name=$(hydra_list "$@")"
      ;;
    full)
      printf '%s\n' "EVALUATION.task_name=null"
      ;;
  esac
}

resolve_mode_overrides() {
  local -a modes=()
  local -a phases=()
  local clean_config=""
  local random_config=""
  local mode
  IFS=',' read -ra modes <<< "${MODES_CSV}"
  for mode in "${modes[@]}"; do
    mode="$(trim "${mode}")"
    case "${mode}" in
      easy)
        phases+=("clean")
        clean_config="demo_clean_rgb_eval"
        ;;
      hard)
        phases+=("random")
        random_config="demo_randomized_no_light_rgb_eval"
        ;;
      clean)
        phases+=("clean")
        clean_config="demo_clean"
        ;;
      random)
        phases+=("random")
        random_config="demo_randomized"
        ;;
      "")
        ;;
      *)
        echo "Unsupported MODES_CSV item: ${mode}. Expected easy,hard,clean,random." >&2
        exit 1
        ;;
    esac
  done
  if [[ "${#phases[@]}" -eq 0 ]]; then
    echo "No modes selected via MODES_CSV=${MODES_CSV}" >&2
    exit 1
  fi
  printf '%s\n' "EVALUATION.phases=$(hydra_list "${phases[@]}")"
  if [[ -n "${clean_config}" ]]; then
    printf '%s\n' "EVALUATION.clean_task_config=${clean_config}"
  fi
  if [[ -n "${random_config}" ]]; then
    printf '%s\n' "EVALUATION.random_task_config=${random_config}"
  fi
}

case "${RUN_MODE}" in
  smoke)
    EPISODES="${EVAL_EPISODES:-${SMOKE_EPISODES}}"
    ;;
  one)
    EPISODES="${EVAL_EPISODES:-${SMOKE_EPISODES}}"
    ;;
  full)
    EPISODES="${EVAL_EPISODES:-${FULL_EPISODES}}"
    ;;
esac

write_runtime_task_configs
TASK_OVERRIDE="$(resolve_task_name_override "$@")"
mapfile -t MODE_OVERRIDES < <(resolve_mode_overrides)

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${RESULT_ROOT}/${RUN_MODE}/${RUN_TS}"
mkdir -p "${OUTPUT_DIR}"

cd "${FASTWAM_DIR}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_DIR}/checkpoints}"

exec "${FASTWAM_PYTHON}" experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  "ckpt=${CKPT_PATH}" \
  "EVALUATION.dataset_stats_path=${STATS_PATH}" \
  "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
  "${TASK_OVERRIDE}" \
  "${MODE_OVERRIDES[@]}" \
  "EVALUATION.eval_num_episodes=${EPISODES}" \
  "EVALUATION.output_dir=${OUTPUT_DIR}" \
  "MULTIRUN.num_gpus=${NUM_GPUS}" \
  "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
