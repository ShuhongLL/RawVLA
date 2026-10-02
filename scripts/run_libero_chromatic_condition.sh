#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_libero_chromatic_condition.sh MODEL FAMILY SETTING THETA_DEG [TRIALS]

Runs one chromatic condition for one model. The server is started once, then all
selected LIBERO suites are evaluated sequentially with independent resume files.

Examples:
  run_libero_chromatic_condition.sh Qwen3-VL-OFT-LIBERO-4in1 uv_whitepoint uv_r0p075 90
  run_libero_chromatic_condition.sh PI05 color_relation rel_s5p5 225 50
EOF
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage
  exit 2
fi

MODEL="$1"
CHROMATIC_FAMILY="$2"
CHROMATIC_SETTING="$3"
CHROMATIC_THETA_DEG="$4"
TRIALS="${5:-${NUM_TRIALS:-50}}"
MAX_TASKS="${MAX_TASKS:--1}"
TASK_CHUNK_SIZE="${TASK_CHUNK_SIZE:-0}"
TASKS_PER_SUITE="${TASKS_PER_SUITE:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$ROOT}"
REPO="${REPO:-$ROOT/starVLA}"
RESULT_ROOT="${RESULT_ROOT:-$DATA_ROOT/results/libero-chromatic-zeroshot}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-$DATA_ROOT/starVLA/playground/Pretrained_models}"
STAR="${STAR:-$PRETRAINED_ROOT/StarVLA}"
BASE="${BASE:-$PRETRAINED_ROOT}"
OPENPI="${OPENPI:-$DATA_ROOT/openpi_converted_protocol}"
QWEN25_BASE="${QWEN25_BASE:-$BASE/Qwen2.5-VL-3B-Instruct}"
QWEN25_FAST_BASE="${QWEN25_FAST_BASE:-$STAR/Qwen2.5-VL-3B-Instruct-Action}"
QWEN3_BASE="${QWEN3_BASE:-$BASE/Qwen3-VL-4B-Instruct}"
COSMOS_BASE="${COSMOS_BASE:-$BASE/nvidia/Cosmos-Predict2-2B-Video2World}"
WAN_BASE="${WAN_BASE:-$BASE/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-$OPENPI/paligemma_tokenizer.model}"
RAWVLA_ACTIVATE="${RAWVLA_ACTIVATE:-}"
GPU="${GPU:-0}"
PORT="${PORT:-18000}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-900}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
SEED="${SEED:-7}"
SUITES_CSV="${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"

IFS=',' read -r -a SUITE_LIST <<<"$SUITES_CSV"

case "$CHROMATIC_FAMILY" in
  uv_whitepoint)
    case "$CHROMATIC_SETTING" in uv_r0p060|uv_r0p075) ;; *) echo "Unsupported uv setting: $CHROMATIC_SETTING" >&2; exit 2 ;; esac
    ;;
  color_relation)
    case "$CHROMATIC_SETTING" in rel_s4p5|rel_s5p5) ;; *) echo "Unsupported relation setting: $CHROMATIC_SETTING" >&2; exit 2 ;; esac
    ;;
  *) echo "Unsupported chromatic family: $CHROMATIC_FAMILY" >&2; exit 2 ;;
esac

case "$CHROMATIC_THETA_DEG" in
  0|45|90|135|180|225|270|315) ;;
  *) echo "CHROMATIC_THETA_DEG must be one of 0,45,90,135,180,225,270,315; got $CHROMATIC_THETA_DEG" >&2; exit 2 ;;
esac

if [[ -z "${PY:-}" && -f "$RAWVLA_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$RAWVLA_ACTIVATE"
fi

if [[ "${RAWVLA_FORCE_IMAGE_PYTHON:-0}" == "1" && -x /usr/local/python3.10.12/bin/python3 ]]; then
  export PATH="/usr/local/python3.10.12/bin:$PATH"
  python() { /usr/local/python3.10.12/bin/python3 "$@"; }
  export -f python
  export RAWVLA_EXTRA_SITE="${RAWVLA_EXTRA_SITE:-}"
  unset CONDA_PREFIX
  unset CONDA_DEFAULT_ENV
  export RAWVLA_PYTHON_MODE=image-python
fi

resolve_python() {
  local explicit="${PY:-${STARVLA_PYTHON:-}}"
  if [[ -n "$explicit" ]]; then
    if [[ "$explicit" != */* ]] && command -v "$explicit" >/dev/null 2>&1; then
      printf '%s\n' "$explicit"
      return 0
    fi
    if [[ -x "$explicit" ]]; then
      printf '%s\n' "$explicit"
      return 0
    fi
    echo "Specified python is not executable: $explicit" >&2
    return 1
  fi

  local -a candidates=()
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("$CONDA_PREFIX/bin/python")
  fi
  candidates+=(
    "/opt/conda/envs/starVLA/bin/python"
    "/usr/local/bin/python"
    "/usr/bin/python3"
  )
  if [[ -n "${RAWVLA_PYTHON_MODE:-}" ]] && command -v python >/dev/null 2>&1; then
    printf '%s\n' python
    return 0
  fi
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  command -v python || command -v python3
}

PY="$(resolve_python)"

checkpoint_for() {
  case "$1" in
    Qwen3-VL-OFT-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_50000_pytorch_model.pt" ;;
    Qwen3-VL-PI-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_100000_pytorch_model.pt" ;;
    WM4A-CosmoPredict-GR00T-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_50000_pytorch_model.pt" ;;
    WM4A-Wan2d2-OFT-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_60000_pytorch_model.pt" ;;
    PI0) echo "$OPENPI/pi0_libero_starvla/bfloat16/model.safetensors" ;;
    PI05|PI0.5) echo "$OPENPI/pi05_libero_starvla/bfloat16/model.safetensors" ;;
    *) echo "Unknown model: $1" >&2; return 1 ;;
  esac
}

server_overrides() {
  case "$1" in
    WM4A-CosmoPredict-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN3_BASE" --config_override framework.qwenvl.attn_implementation=sdpa --config_override "framework.world_model.base_wm=$COSMOS_BASE" ;;
    WM4A-Wan2d2-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN3_BASE" --config_override framework.qwenvl.attn_implementation=sdpa --config_override "framework.world_model.base_wm=$WAN_BASE" ;;
    Qwen3-VL-PI-LIBERO-4in1)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN3_BASE" --config_override framework.qwenvl.attn_implementation=sdpa --config_override framework.action_model.diffusion_model_cfg.use_canonical_forward=false ;;
    Qwen3-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN3_BASE" --config_override framework.qwenvl.attn_implementation=sdpa ;;
    Qwen2.5-VL-FAST-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN25_FAST_BASE" --config_override framework.qwenvl.attn_implementation=sdpa ;;
    Qwen2.5-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN25_BASE" --config_override framework.qwenvl.attn_implementation=sdpa ;;
  esac
}

safe_name() {
  printf '%s' "$1" | tr '/: ' '___'
}

CKPT="$(checkpoint_for "$MODEL")"
MODEL_SAFE="$(safe_name "$MODEL")"
THETA_PAD="$(printf '%03d' "$CHROMATIC_THETA_DEG")"
CONDITION_DIR="$RESULT_ROOT/$MODEL_SAFE/$CHROMATIC_FAMILY/$CHROMATIC_SETTING/theta_${THETA_PAD}"
CONTROL_DIR="$RESULT_ROOT/control"
SERVER_LOG="$CONDITION_DIR/server.log"
JOB_LOG="$CONDITION_DIR/job.log"
mkdir -p "$CONDITION_DIR" "$CONTROL_DIR"

exec > >(tee -a "$JOB_LOG") 2>&1

echo "[$(date -Is)] chromatic condition start model=$MODEL family=$CHROMATIC_FAMILY setting=$CHROMATIC_SETTING theta=$CHROMATIC_THETA_DEG trials=$TRIALS suites=$SUITES_CSV gpu=$GPU port=$PORT"
echo "[$(date -Is)] python=$PY"
echo "[$(date -Is)] checkpoint=$CKPT"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_SKIP_VIDEO="${LIBERO_SKIP_VIDEO:-1}"
export OPENPI_CONVERTED_ROOT="$OPENPI"
export PALIGEMMA_TOKENIZER
unset DEBUG

LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$CONTROL_DIR/libero_config}"
export LIBERO_CONFIG_PATH
if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
  mkdir -p "$LIBERO_CONFIG_PATH"
  "$PY" - "$LIBERO_CONFIG_PATH/config.yaml" "$RESULT_ROOT/libero_datasets" <<'PY'
from pathlib import Path
import sys
import libero

out = Path(sys.argv[1])
datasets = Path(sys.argv[2])
root = Path(libero.__file__).resolve().parent / "libero"
out.write_text(
    "\n".join(
        [
            f"assets: {root / 'assets'}",
            f"bddl_files: {root / 'bddl_files'}",
            f"benchmark_root: {root}",
            f"datasets: {datasets}",
            f"init_states: {root / 'init_files'}",
            "",
        ]
    ),
    encoding="utf-8",
)
PY
fi
echo "[$(date -Is)] LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"

all_done=1
for suite in "${SUITE_LIST[@]}"; do
  suite="${suite//[[:space:]]/}"
  [[ -z "$suite" ]] && continue
  if [[ ! -f "$CONDITION_DIR/$suite/job.done" ]]; then
    all_done=0
  fi
done
if [[ "$all_done" -eq 1 ]]; then
  echo "[$(date -Is)] condition already complete; refreshing summary only"
  "$PY" "$SCRIPT_DIR/summarize_libero_chromatic.py" "$RESULT_ROOT" >"$RESULT_ROOT/summary.txt" || true
  exit 0
fi

wait_server() {
  local pid="$1" log="$2"
  for _ in $(seq 1 "$SERVER_WAIT_ATTEMPTS"); do
    kill -0 "$pid" 2>/dev/null || return 1
    grep -Eq 'server running|server listening' "$log" 2>/dev/null && return 0
    sleep 2
  done
  return 1
}

server_pid=""
kill_server_tree() {
  local pid="${1:-}"
  [[ -z "$pid" ]] && return 0
  kill -0 "$pid" 2>/dev/null || return 0
  echo "[$(date -Is)] stopping server pid=$pid"
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
  local grace="${RAWVLA_SERVER_CLEANUP_GRACE:-30}"
  for _ in $(seq 1 "$grace"); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  echo "[$(date -Is)] force stopping server pid=$pid"
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill_server_tree "$server_pid"
    wait "$server_pid" 2>/dev/null || true
    rm -f "$CONDITION_DIR/server.pid"
    server_pid=""
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  if [[ -f "$CONDITION_DIR/server.pid" ]]; then
    kill_server_tree "$(cat "$CONDITION_DIR/server.pid" 2>/dev/null || true)"
    rm -f "$CONDITION_DIR/server.pid"
  fi
  : >"$SERVER_LOG"
  if [[ "$MODEL" == "PI0" || "$MODEL" == "PI05" || "$MODEL" == "PI0.5" ]]; then
    local pi_model="$MODEL"
    [[ "$pi_model" == "PI0.5" ]] && pi_model="PI05"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/server_starvla_openpi.py \
      --model "$pi_model" --model-source openpi --checkpoint "$CKPT" \
      --tokenizer "$PALIGEMMA_TOKENIZER" --precision bfloat16 --port "$PORT" --idle-timeout -1 \
      >>"$SERVER_LOG" 2>&1 &
  else
    mapfile -t overrides < <(server_overrides "$MODEL")
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" deployment/model_server/server_policy.py \
      --ckpt_path "$CKPT" --port "$PORT" --use_bf16 --idle_timeout -1 \
      "${overrides[@]}" >>"$SERVER_LOG" 2>&1 &
  fi
  server_pid=$!
  echo "$server_pid" >"$CONDITION_DIR/server.pid"
  wait_server "$server_pid" "$SERVER_LOG"
}

run_suite() {
  local suite="$1"
  local suite_dir="$CONDITION_DIR/$suite"
  local eval_log="$suite_dir/eval.log"
  local result_json="$suite_dir/result.json"
  local resume="$suite_dir/episodes.jsonl"
  mkdir -p "$suite_dir/tasks" "$suite_dir/videos"
  if [[ -f "$suite_dir/job.done" ]]; then
    echo "[$(date -Is)] skip completed suite=$suite"
    return 0
  fi
  echo "[$(date -Is)] suite start suite=$suite family=$CHROMATIC_FAMILY setting=$CHROMATIC_SETTING theta=$CHROMATIC_THETA_DEG"
  if [[ "$TASK_CHUNK_SIZE" -gt 0 ]]; then
    local task_start=0
    local task_end=0
    while [[ "$task_start" -lt "$TASKS_PER_SUITE" ]]; do
      task_end=$((task_start + TASK_CHUNK_SIZE))
      if [[ "$task_end" -gt "$TASKS_PER_SUITE" ]]; then
        task_end="$TASKS_PER_SUITE"
      fi
      echo "[$(date -Is)] suite chunk start suite=$suite task_start=$task_start task_end=$task_end"
      if ! start_server; then
        echo "[$(date -Is)] server failed to become ready; see $SERVER_LOG" >&2
        return 1
      fi
      echo "[$(date -Is)] server ready for suite=$suite task_start=$task_start task_end=$task_end"
      set +e
      run_suite_eval "$suite" "$suite_dir" "$eval_log" "$result_json" "$resume" "$task_start" "$task_end"
      local rc=$?
      set -e
      cleanup
      if [[ "$rc" -ne 0 ]]; then
        echo "[$(date -Is)] suite chunk failed suite=$suite task_start=$task_start task_end=$task_end rc=$rc"
        return "$rc"
      fi
      echo "[$(date -Is)] suite chunk done suite=$suite task_start=$task_start task_end=$task_end"
      task_start="$task_end"
    done
    touch "$suite_dir/job.done"
    echo "[$(date -Is)] suite done suite=$suite"
    return 0
  fi
  run_suite_eval "$suite" "$suite_dir" "$eval_log" "$result_json" "$resume" "" ""
}

run_suite_eval() {
  local suite="$1"
  local suite_dir="$2"
  local eval_log="$3"
  local result_json="$4"
  local resume="$5"
  local task_start="$6"
  local task_end="$7"
  local -a openpi_task_args=()
  local -a native_task_args=()
  if [[ -n "$task_start" && -n "$task_end" ]]; then
    openpi_task_args=(--task-start "$task_start" --task-end "$task_end")
    native_task_args=(--args.task-start "$task_start" --args.task-end "$task_end")
  fi
  set +e
  # Older bash/nounset combinations can treat an empty local array expansion as
  # unbound. Disable nounset only while building the eval command line.
  set +u
  if [[ "$MODEL" == "PI0" || "$MODEL" == "PI05" || "$MODEL" == "PI0.5" ]]; then
    local pi_model="$MODEL"
    [[ "$pi_model" == "PI0.5" ]] && pi_model="PI05"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py \
      --model "$pi_model" --model-source openpi --host 127.0.0.1 --port "$PORT" \
      --task-suite "$suite" --num-trials "$TRIALS" --max-tasks "$MAX_TASKS" --resize-size "$RESIZE_SIZE" \
      "${openpi_task_args[@]}" \
      --seed "$SEED" --result-json "$result_json" --resume-path "$resume" \
      --image-mode chromatic --chromatic-family "$CHROMATIC_FAMILY" \
      --chromatic-setting "$CHROMATIC_SETTING" --chromatic-theta-deg "$CHROMATIC_THETA_DEG" \
      --image-quantize-bits 0 2>&1 | tee -a "$eval_log"
  else
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
      --args.pretrained-path "$CKPT" --args.host 127.0.0.1 --args.port "$PORT" \
      --args.task-suite-name "$suite" --args.num-trials-per-task "$TRIALS" --args.max-tasks "$MAX_TASKS" \
      "${native_task_args[@]}" \
      --args.video-out-path "$suite_dir/videos" --args.resume-path "$resume" \
      --args.task-log-dir "$suite_dir/tasks" --args.job-name "${MODEL}_${suite}_${CHROMATIC_FAMILY}_${CHROMATIC_SETTING}_t${THETA_PAD}" \
      --args.image-mode chromatic --args.chromatic-family "$CHROMATIC_FAMILY" \
      --args.chromatic-setting "$CHROMATIC_SETTING" --args.chromatic-theta-deg "$CHROMATIC_THETA_DEG" \
      --args.image-quantize-bits 0 2>&1 | tee -a "$eval_log"
  fi
  local rc=${PIPESTATUS[0]}
  set -u
  set -e
if [[ "$rc" -eq 0 ]]; then
    if [[ "$TASK_CHUNK_SIZE" -le 0 ]]; then
      touch "$suite_dir/job.done"
      echo "[$(date -Is)] suite done suite=$suite"
    fi
  else
    echo "[$(date -Is)] suite failed suite=$suite rc=$rc"
  fi
  return "$rc"
}

if [[ "$TASK_CHUNK_SIZE" -le 0 ]]; then
  if ! start_server; then
    echo "[$(date -Is)] server failed to become ready; see $SERVER_LOG" >&2
    exit 1
  fi
  echo "[$(date -Is)] server ready"
fi

for suite in "${SUITE_LIST[@]}"; do
  suite="${suite//[[:space:]]/}"
  [[ -z "$suite" ]] && continue
  case "$suite" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *) echo "Unsupported suite: $suite" >&2; exit 2 ;;
  esac
  run_suite "$suite"
done

"$PY" "$SCRIPT_DIR/summarize_libero_chromatic.py" "$RESULT_ROOT" >"$RESULT_ROOT/summary.txt" || true
echo "[$(date -Is)] chromatic condition finished"
