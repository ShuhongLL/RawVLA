#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_libero_tonal_condition.sh MODEL TONAL_SETTING TONAL_C [TRIALS]

Runs one tonal-response condition for one model. The server is started once,
then all selected LIBERO suites are evaluated sequentially with independent
resume files.
EOF
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
  exit 2
fi

MODEL="$1"
TONAL_SETTING="$2"
TONAL_C="$3"
TRIALS="${4:-${NUM_TRIALS:-50}}"
MAX_TASKS="${MAX_TASKS:--1}"
TASK_CHUNK_SIZE="${TASK_CHUNK_SIZE:-0}"
TASKS_PER_SUITE="${TASKS_PER_SUITE:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$ROOT}"
REPO="${REPO:-$ROOT/starVLA}"
RESULT_ROOT="${RESULT_ROOT:-$DATA_ROOT/results/libero-tonal-zeroshot}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-$DATA_ROOT/starVLA/playground/Pretrained_models}"
STAR="${STAR:-$PRETRAINED_ROOT/StarVLA}"
BASE="${BASE:-$PRETRAINED_ROOT}"
OPENPI="${OPENPI:-$DATA_ROOT/openpi_converted_protocol}"
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
TONAL_MODE="${TONAL_MODE:-raw_reprocess_isp_tone}"
TONAL_PIVOT="${TONAL_PIVOT:-0.18}"

IFS=',' read -r -a SUITE_LIST <<<"$SUITES_CSV"

case "$TONAL_SETTING" in
  tonal_c0p05|tonal_c0p335|tonal_c0p646|tonal_c2p50|tonal_c5p00|tonal_c12p00) ;;
  *) echo "Unsupported tonal setting: $TONAL_SETTING" >&2; exit 2 ;;
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
  esac
}

safe_name() {
  printf '%s' "$1" | tr '/: ' '___'
}

CKPT="$(checkpoint_for "$MODEL")"
MODEL_SAFE="$(safe_name "$MODEL")"
CONDITION_DIR="$RESULT_ROOT/$MODEL_SAFE/$TONAL_MODE/$TONAL_SETTING"
CONTROL_DIR="$RESULT_ROOT/control"
RUN_TAG="$(safe_name "${RUN_TAG:-${SUITES_CSV}_gpu${GPU}_port${PORT}}")"
SERVER_LOG="$CONDITION_DIR/server.${RUN_TAG}.log"
SERVER_PID="$CONDITION_DIR/server.${RUN_TAG}.pid"
JOB_LOG="$CONDITION_DIR/job.${RUN_TAG}.log"
mkdir -p "$CONDITION_DIR" "$CONTROL_DIR"

: >"$JOB_LOG"
exec > >(tee -a "$JOB_LOG") 2>&1

echo "[$(date -Is)] tonal condition start model=$MODEL setting=$TONAL_SETTING c=$TONAL_C mode=$TONAL_MODE pivot=$TONAL_PIVOT trials=$TRIALS suites=$SUITES_CSV gpu=$GPU port=$PORT"
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
  "$PY" "$SCRIPT_DIR/summarize_libero_tonal.py" "$RESULT_ROOT" >"$RESULT_ROOT/summary.txt" || true
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
    rm -f "$SERVER_PID"
    server_pid=""
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  if [[ -f "$SERVER_PID" ]]; then
    kill_server_tree "$(cat "$SERVER_PID" 2>/dev/null || true)"
    rm -f "$SERVER_PID"
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
  echo "$server_pid" >"$SERVER_PID"
  wait_server "$server_pid" "$SERVER_LOG"
}

run_suite_eval() {
  local suite="$1"
  local suite_dir="$2"
  local eval_log="$3"
  local result_json="$4"
  local resume="$5"
  local task_start="$6"
  local task_end="$7"
  set +e
  if [[ "$MODEL" == "PI0" || "$MODEL" == "PI05" || "$MODEL" == "PI0.5" ]]; then
    local pi_model="$MODEL"
    [[ "$pi_model" == "PI0.5" ]] && pi_model="PI05"
    if [[ -n "$task_start" && -n "$task_end" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py \
        --model "$pi_model" --model-source openpi --host 127.0.0.1 --port "$PORT" \
        --task-suite "$suite" --num-trials "$TRIALS" --max-tasks "$MAX_TASKS" --resize-size "$RESIZE_SIZE" \
        --task-start "$task_start" --task-end "$task_end" \
        --seed "$SEED" --result-json "$result_json" --resume-path "$resume" \
        --image-mode tonal --tonal-mode "$TONAL_MODE" --tonal-setting "$TONAL_SETTING" \
        --tonal-c "$TONAL_C" --tonal-pivot "$TONAL_PIVOT" --image-quantize-bits 0 \
        2>&1 | tee -a "$eval_log"
    else
      CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py \
        --model "$pi_model" --model-source openpi --host 127.0.0.1 --port "$PORT" \
        --task-suite "$suite" --num-trials "$TRIALS" --max-tasks "$MAX_TASKS" --resize-size "$RESIZE_SIZE" \
        --seed "$SEED" --result-json "$result_json" --resume-path "$resume" \
        --image-mode tonal --tonal-mode "$TONAL_MODE" --tonal-setting "$TONAL_SETTING" \
        --tonal-c "$TONAL_C" --tonal-pivot "$TONAL_PIVOT" --image-quantize-bits 0 \
        2>&1 | tee -a "$eval_log"
    fi
  else
    if [[ -n "$task_start" && -n "$task_end" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path "$CKPT" --args.host 127.0.0.1 --args.port "$PORT" \
        --args.task-suite-name "$suite" --args.num-trials-per-task "$TRIALS" --args.max-tasks "$MAX_TASKS" \
        --args.task-start "$task_start" --args.task-end "$task_end" \
        --args.video-out-path "$suite_dir/videos" --args.resume-path "$resume" \
        --args.task-log-dir "$suite_dir/tasks" --args.job-name "${MODEL}_${suite}_${TONAL_SETTING}" \
        --args.image-mode tonal --args.tonal-mode "$TONAL_MODE" --args.tonal-setting "$TONAL_SETTING" \
        --args.tonal-c "$TONAL_C" --args.tonal-pivot "$TONAL_PIVOT" --args.image-quantize-bits 0 \
        2>&1 | tee -a "$eval_log"
    else
      CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path "$CKPT" --args.host 127.0.0.1 --args.port "$PORT" \
        --args.task-suite-name "$suite" --args.num-trials-per-task "$TRIALS" --args.max-tasks "$MAX_TASKS" \
        --args.video-out-path "$suite_dir/videos" --args.resume-path "$resume" \
        --args.task-log-dir "$suite_dir/tasks" --args.job-name "${MODEL}_${suite}_${TONAL_SETTING}" \
        --args.image-mode tonal --args.tonal-mode "$TONAL_MODE" --args.tonal-setting "$TONAL_SETTING" \
        --args.tonal-c "$TONAL_C" --args.tonal-pivot "$TONAL_PIVOT" --args.image-quantize-bits 0 \
        2>&1 | tee -a "$eval_log"
    fi
  fi
  local rc=${PIPESTATUS[0]}
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

chunk_done() {
  local resume="$1"
  local task_start="$2"
  local task_end="$3"
  [[ -f "$resume" ]] || return 1
  "$PY" - "$resume" "$task_start" "$task_end" "$TRIALS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
task_start = int(sys.argv[2])
task_end = int(sys.argv[3])
trials = int(sys.argv[4])
counts = {task_id: 0 for task_id in range(task_start, task_end)}
with path.open("r", encoding="utf-8") as handle:
    for line in handle:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = record.get("task_id")
        if task_id in counts:
            counts[task_id] += 1
missing = [task_id for task_id, count in counts.items() if count < trials]
raise SystemExit(1 if missing else 0)
PY
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
  echo "[$(date -Is)] suite start suite=$suite setting=$TONAL_SETTING c=$TONAL_C"
  if [[ "$TASK_CHUNK_SIZE" -gt 0 ]]; then
    local task_start=0
    local task_end=0
    while [[ "$task_start" -lt "$TASKS_PER_SUITE" ]]; do
      task_end=$((task_start + TASK_CHUNK_SIZE))
      if [[ "$task_end" -gt "$TASKS_PER_SUITE" ]]; then
        task_end="$TASKS_PER_SUITE"
      fi
      if chunk_done "$resume" "$task_start" "$task_end"; then
        echo "[$(date -Is)] suite chunk skip completed suite=$suite task_start=$task_start task_end=$task_end"
        task_start="$task_end"
        continue
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

"$PY" "$SCRIPT_DIR/summarize_libero_tonal.py" "$RESULT_ROOT" >"$RESULT_ROOT/summary.txt" || true
echo "[$(date -Is)] tonal condition finished"
