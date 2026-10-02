#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$CODE_ROOT}"
STARVLA_DIR="${STARVLA_DIR:-${CODE_ROOT}/starVLA}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-${DATA_ROOT}/RoboTwin}"
CKPT_ROOT="${CKPT_ROOT:-${DATA_ROOT}/starvla_robotwin2_checkpoints}"
RESULT_ROOT="${RESULT_ROOT:-${DATA_ROOT}/results/robotwin2-starvla-rgb}"
QWEN3_BASE_VLM="${QWEN3_BASE_VLM:-${DATA_ROOT}/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
RAWVLA_ACTIVATE="${RAWVLA_ACTIVATE:-}"

STARVLA_PYTHON="${STARVLA_PYTHON:-}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-}"

TRIALS="${TRIALS:-50}"
SMOKE_TRIALS="${SMOKE_TRIALS:-1}"
SMOKE_TASKS="${SMOKE_TASKS:-adjust_bottle}"
BASE_PORT="${BASE_PORT:-5694}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-900}"
RUN_MODE="${1:-smoke}"
shift || true
SERVER_CUDA_VISIBLE_DEVICES="${ROBOTWIN_SERVER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
EVAL_CUDA_DEVICE="${ROBOTWIN_EVAL_CUDA_DEVICE:-${SERVER_CUDA_VISIBLE_DEVICES%%,*}}"
CONTROL_NAME="${ROBOTWIN_CONTROL_NAME:-control}"
CONTROL_DIR="${RESULT_ROOT}/${RUN_MODE}/${CONTROL_NAME}"
MASTER_LOG="${CONTROL_DIR}/master.log"
STATUS_FILE="${CONTROL_DIR}/status"
PID_FILE="${CONTROL_DIR}/queue.pid"

TASKS=(
  adjust_bottle
  beat_block_hammer
  blocks_ranking_rgb
  blocks_ranking_size
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_block
  handover_mic
  hanging_mug
  lift_pot
  move_can_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  open_laptop
  open_microwave
  pick_diverse_bottles
  pick_dual_bottles
  place_a2b_left
  place_a2b_right
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_can_basket
  place_cans_plasticbox
  place_container_plate
  place_dual_shoes
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  scan_object
  shake_bottle_horizontally
  shake_bottle
  stack_blocks_three
  stack_blocks_two
  stack_bowls_three
  stack_bowls_two
  stamp_seal
  turn_switch
)

MODE_NAMES=(easy hard)
TASK_CONFIG_SUFFIX="${ROBOTWIN_TASK_CONFIG_SUFFIX:-rgb_eval}"
MODE_CONFIGS=("demo_clean_${TASK_CONFIG_SUFFIX}" "demo_randomized_no_light_${TASK_CONFIG_SUFFIX}")
CKPT_NAMES=(Qwen3-VL-OFT-RoboTwin2-All Qwen3-VL-OFT-Robotwin2)
CKPT_PATHS=(
  "${CKPT_ROOT}/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt"
  "${CKPT_ROOT}/Qwen3-VL-OFT-Robotwin2/checkpoints/steps_40000_pytorch_model.pt"
)

if [[ -n "${ROBOTWIN_CUSTOM_CKPT_NAME:-}" && -n "${ROBOTWIN_CUSTOM_CKPT_PATH:-}" ]]; then
  CKPT_NAMES=("${ROBOTWIN_CUSTOM_CKPT_NAME}")
  CKPT_PATHS=("${ROBOTWIN_CUSTOM_CKPT_PATH}")
fi

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/run_robotwin2_starvla_rgb_eval.sh [smoke|full|one] [task...]

Modes:
  smoke  Run both checkpoints on easy+hard for SMOKE_TASKS, with SMOKE_TRIALS.
  full   Run both checkpoints on all 50 RoboTwin 2.0 tasks, with TRIALS=50.
  one    Run both checkpoints on easy+hard for the provided task list.

Useful env:
  TRIALS=50 SMOKE_TRIALS=1 SMOKE_TASKS=adjust_bottle
  CUDA_VISIBLE_DEVICES=0,1
  CKPTS_CSV=Qwen3-VL-OFT-RoboTwin2-All MODES_CSV=easy TASKS_CSV=adjust_bottle,open_laptop
  RAWVLA_QWEN_SHARD=1 RAWVLA_QWEN_DEVICE_MAP=balanced_low_0
  RAWVLA_QWEN_MAX_MEMORY=0:8GiB,1:14GiB
  QWEN3_BASE_VLM=/path/to/Qwen3-VL-4B-Instruct
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

require_executable() {
  local command_or_path="$1"
  if [[ "${command_or_path}" == */* ]]; then
    if [[ ! -x "${command_or_path}" ]]; then
      echo "Missing executable: ${command_or_path}" >&2
      exit 1
    fi
  elif ! command -v "${command_or_path}" >/dev/null 2>&1; then
    echo "Missing executable command: ${command_or_path}" >&2
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

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "${value}"
}

join_arr() {
  local sep="$1"; shift
  local out=""
  local item
  for item in "$@"; do
    out="${out:+${out}${sep}}${item}"
  done
  printf '%s' "${out}"
}

setup_runtime_python() {
  if [[ -z "${STARVLA_PYTHON:-}" && -f "${RAWVLA_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${RAWVLA_ACTIVATE}"
  fi

  if [[ "${RAWVLA_FORCE_IMAGE_PYTHON:-0}" == "1" && -x /usr/local/python3.10.12/bin/python3 ]]; then
    export PATH="/usr/local/python3.10.12/bin:${PATH}"
    python() { /usr/local/python3.10.12/bin/python3 "$@"; }
    export -f python
    export RAWVLA_EXTRA_SITE="${RAWVLA_EXTRA_SITE:-}"
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    export RAWVLA_PYTHON_MODE=image-python
  fi
}

find_conda_python() {
  local env_name="$1"
  local -a env_names=("${env_name}")
  local lower_env="${env_name,,}"
  if [[ "${lower_env}" != "${env_name}" ]]; then
    env_names+=("${lower_env}")
  fi
  if [[ "${env_name}" == "robotwin" ]]; then
    env_names+=(RoboTwin robotwin2)
  fi

  local -a search_dirs=()
  if [[ -n "${CONDA_EXE:-}" ]]; then
    search_dirs+=("$(dirname "$(dirname "${CONDA_EXE}")")/envs")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    search_dirs+=("$(dirname "${CONDA_PREFIX}")")
  fi
  search_dirs+=(
    "${HOME}/anaconda3/envs"
    "${HOME}/miniconda3/envs"
    "${HOME}/miniforge3/envs"
    "/opt/conda/envs"
  )

  local base name py
  for base in "${search_dirs[@]}"; do
    for name in "${env_names[@]}"; do
      for py in "${base}/${name}/bin/python" "${base}/${name}/bin/python3.10" "${base}/${name}/bin/python3"; do
        if [[ -x "${py}" ]]; then
          printf '%s\n' "${py}"
          return 0
        fi
      done
    done
  done
  if [[ "${env_name}" == "starVLA" ]]; then
    for py in /usr/local/python3.10.12/bin/python3 /usr/local/bin/python /usr/bin/python3; do
      if [[ -x "${py}" ]]; then
        printf '%s\n' "${py}"
        return 0
      fi
    done
  fi
  echo "Cannot find Python for conda env '${env_name}'." >&2
  echo "Searched: ${search_dirs[*]}" >&2
  return 1
}

resolve_python() {
  local explicit_path="$1"
  local env_name="$2"
  if [[ -n "${explicit_path}" ]]; then
    if [[ ! -x "${explicit_path}" ]]; then
      echo "Specified python not executable: ${explicit_path}" >&2
      return 1
    fi
    printf '%s\n' "${explicit_path}"
    return 0
  fi
  if [[ "${env_name}" == "starVLA" && -n "${RAWVLA_PYTHON_MODE:-}" ]] && command -v python >/dev/null 2>&1; then
    printf '%s\n' python
    return 0
  fi
  find_conda_python "${env_name}"
}

csv_contains_or_empty() {
  local csv="$1"
  local needle="$2"
  local -a items=()
  local item
  if [[ -z "${csv}" ]]; then
    return 0
  fi
  IFS=',' read -ra items <<< "${csv}"
  for item in "${items[@]}"; do
    item="$(trim "${item}")"
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tuln 2>/dev/null | grep -q ":${port}[[:space:]]"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
  else
    "${STARVLA_PYTHON}" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',${port})); s.close()" 2>/dev/null
  fi
}

wait_for_server() {
  local port="$1"
  local timeout_s="$2"
  local elapsed=0
  while (( elapsed < timeout_s )); do
    if port_in_use "${port}"; then
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  return 1
}

kill_tree() {
  local pid="$1"
  local sig="${2:-TERM}"
  local child
  for child in $(ps -o pid= --ppid "${pid}" 2>/dev/null); do
    kill_tree "${child}" "${sig}"
  done
  kill -"${sig}" "${pid}" 2>/dev/null || true
}

write_runtime_task_configs() {
  local task_config_dir="${ROBOTWIN_PATH}/task_config"
  require_file "${task_config_dir}/demo_clean.yml"
  require_file "${task_config_dir}/demo_randomized.yml"

  sed \
    -e 's/^eval_video_log:.*/eval_video_log: false/' \
    "${task_config_dir}/demo_clean.yml" > "${task_config_dir}/${MODE_CONFIGS[0]}.yml"
  cat >> "${task_config_dir}/${MODE_CONFIGS[0]}.yml" <<EOF
hdr_raw_white_level: ${ROBOTWIN_RAW_WHITE_LEVEL:-3.5}
raw_sensor_noise_enabled: ${ROBOTWIN_RAW_SENSOR_NOISE_ENABLED:-true}
raw_sensor_noise_seed: ${ROBOTWIN_RAW_SENSOR_NOISE_SEED:-20260824}
raw_sensor_shot_noise: ${ROBOTWIN_RAW_SENSOR_SHOT_NOISE:-0.000025}
raw_sensor_read_noise: ${ROBOTWIN_RAW_SENSOR_READ_NOISE:-0.0000000390625}
need_plan: ${ROBOTWIN_NEED_PLAN:-false}
EOF

  sed \
    -e 's/^[[:space:]]*random_light:.*/  random_light: false/' \
    -e 's/^[[:space:]]*crazy_random_light_rate:.*/  crazy_random_light_rate: 0/' \
    -e 's/^eval_video_log:.*/eval_video_log: false/' \
    "${task_config_dir}/demo_randomized.yml" > "${task_config_dir}/${MODE_CONFIGS[1]}.yml"
  cat >> "${task_config_dir}/${MODE_CONFIGS[1]}.yml" <<EOF
hdr_raw_white_level: ${ROBOTWIN_RAW_WHITE_LEVEL:-3.5}
raw_sensor_noise_enabled: ${ROBOTWIN_RAW_SENSOR_NOISE_ENABLED:-true}
raw_sensor_noise_seed: ${ROBOTWIN_RAW_SENSOR_NOISE_SEED:-20260824}
raw_sensor_shot_noise: ${ROBOTWIN_RAW_SENSOR_SHOT_NOISE:-0.000025}
raw_sensor_read_noise: ${ROBOTWIN_RAW_SENSOR_READ_NOISE:-0.0000000390625}
need_plan: ${ROBOTWIN_NEED_PLAN:-false}
EOF
}

resolve_tasks() {
  local -a selected=()
  local item
  if [[ -n "${TASKS_CSV:-}" ]]; then
    IFS=',' read -ra selected <<< "${TASKS_CSV}"
  elif [[ "${RUN_MODE}" == "full" ]]; then
    selected=("${TASKS[@]}")
  elif [[ "${RUN_MODE}" == "smoke" ]]; then
    IFS=',' read -ra selected <<< "${SMOKE_TASKS}"
  else
    if (( $# == 0 )); then
      echo "Mode 'one' needs at least one task." >&2
      exit 1
    fi
    selected=("$@")
  fi
  for item in "${selected[@]}"; do
    item="$(trim "${item}")"
    if [[ -n "${item}" ]]; then
      printf '%s\n' "${item}"
    fi
  done
}

run_eval_task() {
  local ckpt_name="$1"
  local ckpt_path="$2"
  local mode_name="$3"
  local task_config="$4"
  local task="$5"
  local port="$6"
  local trials="$7"
  local out_dir="$8"
  local task_safe="${task//\//_}"
  local log_file="${out_dir}/${task_safe}.log"
  local done_file="${out_dir}/${task_safe}.DONE"
  local failed_file="${out_dir}/${task_safe}.FAILED"

  if [[ -f "${done_file}" ]]; then
    echo "[$(date -Is)] skip done ckpt=${ckpt_name} mode=${mode_name} task=${task}"
    return 0
  fi
  rm -f "${failed_file}"

  echo "[$(date -Is)] start ckpt=${ckpt_name} mode=${mode_name} task=${task} trials=${trials}"
  (
    if [[ "${RAWVLA_PYTHON_MODE:-}" == "image-python" ]]; then
      if [[ "${RAWVLA_FORCE_IMAGE_ROBOTWIN_PYTHON:-1}" == "1" ]]; then
        export ROBOTWIN_PYTHON=python
      fi
      local robotwin_env_root="${RAWVLA_ROBOTWIN_ENV_ROOT:-${CONDA_PREFIX:-}}"
      local robotwin_image_overlay="${RAWVLA_ROBOTWIN_IMAGE_OVERLAY:-}"
      local control_slug="${ROBOTWIN_CONTROL_NAME:-default}"
      control_slug="${control_slug//\//_}"
      control_slug="${control_slug//[^A-Za-z0-9_.-]/_}"
      local robotwin_torch_extensions="${RAWVLA_ROBOTWIN_TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/rawvla_torch_extensions_robotwin_image_py310/${control_slug}}"
      local robotwin_site="${robotwin_env_root}/lib/python3.10/site-packages"
      local robotwin_lib="${robotwin_env_root}/lib"
      local robotwin_curobo_src="${ROBOTWIN_PATH}/envs/curobo/src"
      local sapien_oidn="${robotwin_site}/sapien/oidn_library"
      local sapien_vulkan="${robotwin_site}/sapien/vulkan_library"
      local sapien_libs="${robotwin_site}/sapien.libs"
      local embreex_libs="${robotwin_site}/embreex.libs"
      if [[ -d "${robotwin_site}" ]]; then
        export PYTHONPATH="${robotwin_site}:${PYTHONPATH:-}"
      fi
      if [[ -d "${robotwin_image_overlay}" ]]; then
        export PYTHONPATH="${robotwin_image_overlay}:${PYTHONPATH:-}"
      fi
      if [[ -d "${robotwin_curobo_src}" ]]; then
        export PYTHONPATH="${robotwin_curobo_src}:${PYTHONPATH:-}"
      fi
      mkdir -p "${robotwin_torch_extensions}"
      export TORCH_EXTENSIONS_DIR="${robotwin_torch_extensions}"
      if [[ "${RAWVLA_ROBOTWIN_ADD_ENV_LIB:-0}" == "1" && -d "${robotwin_lib}" ]]; then
        export LD_LIBRARY_PATH="${robotwin_lib}:${LD_LIBRARY_PATH:-}"
      fi
      if [[ -d "${sapien_oidn}" ]]; then
        export LD_LIBRARY_PATH="${sapien_oidn}:${LD_LIBRARY_PATH:-}"
      fi
      if [[ -d "${sapien_vulkan}" ]]; then
        export LD_LIBRARY_PATH="${sapien_vulkan}:${LD_LIBRARY_PATH:-}"
      fi
      if [[ -d "${sapien_libs}" ]]; then
        export LD_LIBRARY_PATH="${sapien_libs}:${LD_LIBRARY_PATH:-}"
      fi
      if [[ -d "${embreex_libs}" ]]; then
        export LD_LIBRARY_PATH="${embreex_libs}:${LD_LIBRARY_PATH:-}"
      fi
    fi
    export ROBOTWIN_PATH
    export ROBOTWIN_PYTHON
    export STARVLA_PYTHON
    export ROBOTWIN_TEST_NUM="${trials}"
    export ROBOTWIN_POLICY_HOST="${POLICY_SERVER_HOST:-127.0.0.1}"
    export ROBOTWIN_POLICY_PORT="${port}"
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
    cd "${STARVLA_DIR}/examples/simBenchmarks/Robotwin/eval_files"
    bash ./eval.sh "${task}" "${task_config}" "${ckpt_name}_${mode_name}_rgb" "${START_EPISODE:-0}" "${EVAL_CUDA_DEVICE}" "${ckpt_path}" "${port}" "${POLICY_SERVER_HOST:-127.0.0.1}"
  ) 2>&1 | tee "${log_file}"
  local rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    printf 'done %s\n' "$(date -Is)" > "${done_file}"
    echo "[$(date -Is)] done ckpt=${ckpt_name} mode=${mode_name} task=${task}"
  else
    printf 'failed %s rc=%s\n' "$(date -Is)" "${rc}" > "${failed_file}"
    echo "[$(date -Is)] failed rc=${rc} ckpt=${ckpt_name} mode=${mode_name} task=${task}"
  fi
  return "${rc}"
}

run_checkpoint() {
  local ckpt_name="$1"
  local ckpt_path="$2"
  local trials="$3"
  local -a selected_tasks=("${@:4}")
  local port="$BASE_PORT"
  local timestamp
  local run_dir
  local server_log
  local server_pid=""
  local mode_idx
  local task
  local failures=0

  while port_in_use "${port}"; do
    port=$((port + 1))
  done
  BASE_PORT=$((port + 1))

  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_dir="${RESULT_ROOT}/${RUN_MODE}/${ckpt_name}"
  mkdir -p "${run_dir}/control"
  server_log="${run_dir}/control/server_${timestamp}.log"
  local all_done=1
  for mode_idx in "${!MODE_NAMES[@]}"; do
    local mode_name_probe="${MODE_NAMES[$mode_idx]}"
    if ! csv_contains_or_empty "${MODES_CSV:-}" "${mode_name_probe}"; then
      continue
    fi
    local mode_dir_probe="${run_dir}/${mode_name_probe}"
    for task in "${selected_tasks[@]}"; do
      local task_safe_probe="${task//\//_}"
      if [[ "${ROBOTWIN_SEEDS_IN_SEPARATE_PROCESSES:-0}" == "1" ]]; then
        local -a probe_seeds=()
        IFS=',' read -ra probe_seeds <<< "${ROBOTWIN_EVAL_SEEDS:-}"
        local probe_seed
        if (( ${#probe_seeds[@]} < trials )); then
          all_done=0
          break
        fi
        for probe_seed in "${probe_seeds[@]:0:trials}"; do
          probe_seed="$(trim "${probe_seed}")"
          if [[ ! -f "${mode_dir_probe}/seed_$(printf '%02d' "${probe_seed}")/${task_safe_probe}.DONE" ]]; then
            all_done=0
            break
          fi
        done
      elif [[ ! -f "${mode_dir_probe}/${task_safe_probe}.DONE" ]]; then
        all_done=0
        break
      fi
    done
    if [[ "${all_done}" -eq 0 ]]; then
      break
    fi
  done
  if [[ "${all_done}" -eq 1 ]]; then
    echo "[$(date -Is)] skip checkpoint complete ckpt=${ckpt_name}"
    return 0
  fi

  cleanup_server() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      kill_tree "${server_pid}" TERM
      sleep 2
      if kill -0 "${server_pid}" 2>/dev/null; then
        kill_tree "${server_pid}" KILL
      fi
      wait "${server_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup_server RETURN

  echo "[INFO] Starting server ckpt=${ckpt_name} port=${port}"
  (
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export RAWVLA_QWEN_SHARD="${RAWVLA_QWEN_SHARD:-1}"
    export RAWVLA_QWEN_DEVICE_MAP="${RAWVLA_QWEN_DEVICE_MAP:-balanced_low_0}"
    export RAWVLA_QWEN_MAX_MEMORY_PER_GPU="${RAWVLA_QWEN_MAX_MEMORY_PER_GPU:-20GiB}"
    export RAWVLA_QWEN_ACTION_DEVICE="${RAWVLA_QWEN_ACTION_DEVICE:-cuda:0}"
    export RAWVLA_EMPTY_CACHE_EACH_STEP="${RAWVLA_EMPTY_CACHE_EACH_STEP:-1}"
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    export STARVLA_PYTHON
    export CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES}"
    export PYTHONPATH="${STARVLA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    local -a server_config_args=(
      --config_override "framework.qwenvl.base_vlm=${QWEN3_BASE_VLM}"
    )
    if [[ -n "${ROBOTWIN_SERVER_CONFIG_OVERRIDES:-}" ]]; then
      local -a extra_server_overrides=()
      local extra_server_override
      IFS=';' read -ra extra_server_overrides <<< "${ROBOTWIN_SERVER_CONFIG_OVERRIDES}"
      for extra_server_override in "${extra_server_overrides[@]}"; do
        extra_server_override="$(trim "${extra_server_override}")"
        if [[ -n "${extra_server_override}" ]]; then
          server_config_args+=(--config_override "${extra_server_override}")
        fi
      done
    fi
    cd "${STARVLA_DIR}"
    "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
      --ckpt_path "${ckpt_path}" \
      --port "${port}" \
      --use_bf16 \
      --idle_timeout -1 \
      "${server_config_args[@]}"
  ) > "${server_log}" 2>&1 &
  server_pid=$!

  if ! wait_for_server "${port}" "${SERVER_TIMEOUT}"; then
    echo "[ERROR] Server did not become ready for ${ckpt_name}; see ${server_log}" >&2
    cleanup_server
    trap - RETURN
    return 1
  fi

  for mode_idx in "${!MODE_NAMES[@]}"; do
    local mode_name="${MODE_NAMES[$mode_idx]}"
    if ! csv_contains_or_empty "${MODES_CSV:-}" "${mode_name}"; then
      continue
    fi
    local task_config="${MODE_CONFIGS[$mode_idx]}"
    local mode_dir="${run_dir}/${mode_name}"
    mkdir -p "${mode_dir}"
    printf '%s\n' "${selected_tasks[@]}" > "${mode_dir}/tasks.txt"
    for task in "${selected_tasks[@]}"; do
      if [[ "${ROBOTWIN_SEEDS_IN_SEPARATE_PROCESSES:-0}" == "1" ]]; then
        local seed_value
        local -a isolated_seeds=()
        IFS=',' read -ra isolated_seeds <<< "${ROBOTWIN_EVAL_SEEDS:?Set ROBOTWIN_EVAL_SEEDS for isolated seed processes}"
        if (( ${#isolated_seeds[@]} < trials )); then
          echo "ROBOTWIN_EVAL_SEEDS has ${#isolated_seeds[@]} values but trials=${trials}" >&2
          failures=$((failures + 1))
          continue
        fi
        for seed_value in "${isolated_seeds[@]:0:trials}"; do
          seed_value="$(trim "${seed_value}")"
          export ROBOTWIN_EVAL_SEEDS="${seed_value}"
          export START_EPISODE="${seed_value}"
          local seed_dir="${mode_dir}/seed_$(printf '%02d' "${seed_value}")"
          mkdir -p "${seed_dir}"
          if ! run_eval_task "${ckpt_name}" "${ckpt_path}" "${mode_name}" "${task_config}" "${task}" "${port}" 1 "${seed_dir}"; then
            failures=$((failures + 1))
          fi
        done
      elif ! run_eval_task "${ckpt_name}" "${ckpt_path}" "${mode_name}" "${task_config}" "${task}" "${port}" "${trials}" "${mode_dir}"; then
        failures=$((failures + 1))
      fi
    done
  done

  "${CODE_ROOT}/scripts/summarize_robotwin2_eval.py" "${RESULT_ROOT}/${RUN_MODE}" > "${run_dir}/summary.txt" || true
  cleanup_server
  trap - RETURN
  return "${failures}"
}

main() {
  mkdir -p "${CONTROL_DIR}"
  exec > >(tee -a "${MASTER_LOG}") 2>&1
  printf '%s\n' "running" > "${STATUS_FILE}"
  echo "$$" > "${PID_FILE}"
  trap 'rc=$?; rm -f "${PID_FILE}"; if [[ $rc -eq 0 ]]; then printf "%s\n" done > "${STATUS_FILE}"; else printf "%s\n" completed_with_failures > "${STATUS_FILE}"; fi; echo "[$(date -Is)] robotwin2 queue exit rc=$rc status=$(cat "${STATUS_FILE}")"' EXIT

  setup_runtime_python
  STARVLA_PYTHON="$(resolve_python "${STARVLA_PYTHON:-}" "${ROBOTWIN_STARVLA_ENV:-starVLA}")"
  ROBOTWIN_PYTHON="$(resolve_python "${ROBOTWIN_PYTHON:-}" "${ROBOTWIN_ENV:-robotwin}")"
  export STARVLA_PYTHON ROBOTWIN_PYTHON

  require_dir "${STARVLA_DIR}"
  require_dir "${ROBOTWIN_PATH}"
  require_file "${STARVLA_DIR}/examples/simBenchmarks/Robotwin/eval_files/eval.sh"
  require_file "${STARVLA_DIR}/examples/simBenchmarks/Robotwin/eval_files/run_policy_server.sh"
  require_executable "${STARVLA_PYTHON}"
  require_executable "${ROBOTWIN_PYTHON}"
  require_dir "${QWEN3_BASE_VLM}"

  local ckpt_path
  for ckpt_path in "${CKPT_PATHS[@]}"; do
    require_file "${ckpt_path}"
  done

  write_runtime_task_configs

  mapfile -t selected_tasks < <(resolve_tasks "$@")
  if (( ${#selected_tasks[@]} == 0 )); then
    echo "No tasks selected." >&2
    exit 1
  fi
  local trials="${TRIALS}"
  if [[ "${RUN_MODE}" == "smoke" ]]; then
    trials="${SMOKE_TRIALS}"
  fi

  mkdir -p "${RESULT_ROOT}/${RUN_MODE}"
  {
    printf 'checkpoint\tmode\ttask\ttrials\n'
    local item_ckpt item_mode item_task
    for item_ckpt in "${CKPT_NAMES[@]}"; do
      if ! csv_contains_or_empty "${CKPTS_CSV:-}" "${item_ckpt}"; then
        continue
      fi
      for item_mode in "${MODE_NAMES[@]}"; do
        if ! csv_contains_or_empty "${MODES_CSV:-}" "${item_mode}"; then
          continue
        fi
        for item_task in "${selected_tasks[@]}"; do
          printf '%s\t%s\t%s\t%s\n' "${item_ckpt}" "${item_mode}" "${item_task}" "${trials}"
        done
      done
    done
  } > "${CONTROL_DIR}/items.tsv"

  echo "[$(date -Is)] robotwin2 queue start pid=$$"
  echo "[INFO] run_mode=${RUN_MODE} trials=${trials}"
  echo "[INFO] robotwin=${ROBOTWIN_PATH}"
  echo "[INFO] qwen3_base_vlm=${QWEN3_BASE_VLM}"
  echo "[INFO] starvla_python=${STARVLA_PYTHON} robotwin_python=${ROBOTWIN_PYTHON} rawvla_python_mode=${RAWVLA_PYTHON_MODE:-unset}"
  echo "[INFO] server_cuda_visible_devices=${SERVER_CUDA_VISIBLE_DEVICES} eval_cuda_device=${EVAL_CUDA_DEVICE}"
  echo "[INFO] ckpts_csv=${CKPTS_CSV:-<all>} modes_csv=${MODES_CSV:-<all>} tasks_csv=${TASKS_CSV:-<default>}"
  echo "[INFO] tasks (${#selected_tasks[@]}): $(join_arr ', ' "${selected_tasks[@]}")"

  local idx
  local failures=0
  for idx in "${!CKPT_NAMES[@]}"; do
    if ! csv_contains_or_empty "${CKPTS_CSV:-}" "${CKPT_NAMES[$idx]}"; then
      continue
    fi
    if ! run_checkpoint "${CKPT_NAMES[$idx]}" "${CKPT_PATHS[$idx]}" "${trials}" "${selected_tasks[@]}"; then
      failures=$((failures + 1))
    fi
  done

  "${CODE_ROOT}/scripts/summarize_robotwin2_eval.py" "${RESULT_ROOT}/${RUN_MODE}" | tee "${RESULT_ROOT}/${RUN_MODE}/summary.txt"
  if [[ "${failures}" -ne 0 ]]; then
    return 1
  fi
}

main "$@"
