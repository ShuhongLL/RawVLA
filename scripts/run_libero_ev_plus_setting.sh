#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_libero_ev_plus_setting.sh MODEL SUITE [TRIALS]

Runs one EV setting group on one machine with one policy server:
  EV=+2/raw_direct
  EV=+2/rgb_recovered
  EV=+4/raw_direct
  EV=+4/rgb_recovered

Each phase has independent logs, resume ledger, and completion marker.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

MODEL="$1"
SUITE="$2"
TRIALS="${3:-${NUM_TRIALS:-50}}"
MAX_TASKS="${MAX_TASKS:--1}"
EVS="${EVS:-2 4}"
REPS="${REPS:-raw_direct rgb_recovered}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$ROOT}"
REPO="${REPO:-$ROOT/starVLA}"
RESULT_ROOT="${RESULT_ROOT:-$DATA_ROOT/results/libero-float32-ev-plus}"
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
TASK_CHUNK_SIZE="${TASK_CHUNK_SIZE:-0}"
TASKS_PER_SUITE="${TASKS_PER_SUITE:-10}"
EPISODE_CHUNK_SIZE="${RAWVLA_EPISODE_CHUNK_SIZE:-${EPISODE_CHUNK_SIZE:-0}}"

if [[ -z "${PY:-}" && -f "$RAWVLA_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$RAWVLA_ACTIVATE"
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
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local conda_base
    conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
    candidates+=("$conda_base/envs/starVLA/bin/python" "$conda_base/envs/starvla/bin/python")
  fi
  candidates+=(
    "$HOME/anaconda3/envs/starVLA/bin/python"
    "$HOME/anaconda3/envs/starvla/bin/python"
    "$HOME/miniconda3/envs/starVLA/bin/python"
    "$HOME/miniconda3/envs/starvla/bin/python"
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
echo "[$(date -Is)] python=$PY"

case "$SUITE" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *) echo "Unsupported suite: $SUITE" >&2; exit 2 ;;
esac

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
SETTING_DIR="$RESULT_ROOT/$MODEL_SAFE/$SUITE"
CONTROL_DIR="$RESULT_ROOT/control"
SERVER_LOG="$SETTING_DIR/server.log"
JOB_LOG="$SETTING_DIR/job.log"
mkdir -p "$SETTING_DIR" "$CONTROL_DIR"

exec > >(tee -a "$JOB_LOG") 2>&1

echo "[$(date -Is)] setting start model=$MODEL suite=$SUITE evs=[$EVS] reps=[$REPS] phases=[${RAWVLA_EV_PHASES:-}] trials=$TRIALS max_tasks=$MAX_TASKS gpu=$GPU port=$PORT task_chunk_size=$TASK_CHUNK_SIZE episode_chunk_size=$EPISODE_CHUNK_SIZE"
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
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup EXIT INT TERM

cleanup_stale_server() {
  cleanup
  if [[ -f "$SETTING_DIR/server.pid" ]]; then
    local old_pid
    old_pid="$(cat "$SETTING_DIR/server.pid" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]]; then
      kill "$old_pid" 2>/dev/null || true
      wait "$old_pid" 2>/dev/null || true
    fi
    rm -f "$SETTING_DIR/server.pid"
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  fi
  sleep "${RAWVLA_PORT_CLEANUP_GRACE:-2}"
}

start_server() {
  cleanup_stale_server
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
  echo "$server_pid" >"$SETTING_DIR/server.pid"
  if wait_server "$server_pid" "$SERVER_LOG"; then
    return 0
  fi
  cleanup_stale_server
  return 1
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
seen = {task_id: set() for task_id in range(task_start, task_end)}
with path.open("r", encoding="utf-8") as handle:
    for line in handle:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = record.get("task_id")
        episode_idx = record.get("episode_idx")
        if task_id in seen and isinstance(episode_idx, int):
            seen[task_id].add(episode_idx)
missing = [task_id for task_id, episodes in seen.items() if len(episodes) < trials]
raise SystemExit(1 if missing else 0)
PY
}

run_phase() {
  local ev="$1"
  local representation="$2"
  local phase_dir="$SETTING_DIR/ev_${ev}/$representation"
  local eval_log="$phase_dir/eval.log"
  local result_json="$phase_dir/result.json"
  local resume="$phase_dir/episodes.jsonl"
  mkdir -p "$phase_dir/tasks" "$phase_dir/videos"
  if [[ -f "$phase_dir/job.done" ]]; then
    echo "[$(date -Is)] skip completed phase ev=$ev representation=$representation"
    return 0
  fi
  echo "[$(date -Is)] phase start ev=$ev representation=$representation"
  if [[ "$TASK_CHUNK_SIZE" -gt 0 ]]; then
    local task_start=0
    local task_end=0
    while [[ "$task_start" -lt "$TASKS_PER_SUITE" ]]; do
      task_end=$((task_start + TASK_CHUNK_SIZE))
      if [[ "$task_end" -gt "$TASKS_PER_SUITE" ]]; then
        task_end="$TASKS_PER_SUITE"
      fi
      echo "[$(date -Is)] phase chunk start ev=$ev representation=$representation task_start=$task_start task_end=$task_end"
      if ! start_server; then
        echo "[$(date -Is)] server failed to become ready; see $SERVER_LOG" >&2
        return 1
      fi
      echo "[$(date -Is)] server ready for ev=$ev representation=$representation task_start=$task_start task_end=$task_end"
      set +e
      run_phase_eval "$ev" "$representation" "$phase_dir" "$eval_log" "$result_json" "$resume" "$task_start" "$task_end"
      local rc=$?
      set -e
      cleanup
      if [[ "$rc" -ne 0 ]]; then
        echo "[$(date -Is)] phase chunk failed ev=$ev representation=$representation task_start=$task_start task_end=$task_end rc=$rc"
        return "$rc"
      fi
      if [[ "$EPISODE_CHUNK_SIZE" -gt 0 ]] && ! chunk_done "$resume" "$task_start" "$task_end"; then
        echo "[$(date -Is)] phase chunk partial ev=$ev representation=$representation task_start=$task_start task_end=$task_end; restarting server"
        continue
      fi
      echo "[$(date -Is)] phase chunk done ev=$ev representation=$representation task_start=$task_start task_end=$task_end"
      task_start="$task_end"
    done
    touch "$phase_dir/job.done"
    echo "[$(date -Is)] phase done ev=$ev representation=$representation"
    return 0
  fi
  run_phase_eval "$ev" "$representation" "$phase_dir" "$eval_log" "$result_json" "$resume" "" ""
}

run_phase_eval() {
  local ev="$1"
  local representation="$2"
  local phase_dir="$3"
  local eval_log="$4"
  local result_json="$5"
  local resume="$6"
  local task_start="$7"
  local task_end="$8"
  local -a openpi_task_args=()
  local -a native_task_args=()
  if [[ -n "$task_start" && -n "$task_end" ]]; then
    openpi_task_args=(--task-start "$task_start" --task-end "$task_end")
    native_task_args=(--args.task-start "$task_start" --args.task-end "$task_end")
  fi
  if [[ "$EPISODE_CHUNK_SIZE" -gt 0 ]]; then
    native_task_args+=(--args.max-new-episodes-per-run "$EPISODE_CHUNK_SIZE")
  fi
  export STARVLA_IMAGE_AUDIT_PATH="$phase_dir/image_audit.jsonl"
  rm -f "$STARVLA_IMAGE_AUDIT_PATH"
  set +e
  set +u
  if [[ "$MODEL" == "PI0" || "$MODEL" == "PI05" || "$MODEL" == "PI0.5" ]]; then
    local pi_model="$MODEL"
    [[ "$pi_model" == "PI0.5" ]] && pi_model="PI05"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py \
      --model "$pi_model" --model-source openpi --host 127.0.0.1 --port "$PORT" \
      --task-suite "$SUITE" --num-trials "$TRIALS" --max-tasks "$MAX_TASKS" --resize-size "$RESIZE_SIZE" \
      "${openpi_task_args[@]}" \
      --seed "$SEED" --result-json "$result_json" --resume-path "$resume" \
      --image-mode ev_float32 --ev-representation "$representation" --exposure-ev "$ev" \
      --image-quantize-bits 0 2>&1 | tee -a "$eval_log"
  else
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
      --args.pretrained-path "$CKPT" --args.host 127.0.0.1 --args.port "$PORT" \
      --args.task-suite-name "$SUITE" --args.num-trials-per-task "$TRIALS" --args.max-tasks "$MAX_TASKS" \
      "${native_task_args[@]}" \
      --args.video-out-path "$phase_dir/videos" --args.resume-path "$resume" \
      --args.task-log-dir "$phase_dir/tasks" --args.job-name "${MODEL}_${SUITE}_ev${ev}_${representation}" \
      --args.image-mode ev_float32 --args.ev-representation "$representation" \
      --args.exposure-ev "$ev" --args.image-quantize-bits 0 2>&1 | tee -a "$eval_log"
  fi
  local rc=${PIPESTATUS[0]}
  set -u
  set -e
  if [[ "$rc" -eq 0 ]]; then
    if [[ "$TASK_CHUNK_SIZE" -le 0 ]]; then
      touch "$phase_dir/job.done"
      echo "[$(date -Is)] phase done ev=$ev representation=$representation"
    fi
  else
    echo "[$(date -Is)] phase failed ev=$ev representation=$representation rc=$rc"
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

if [[ -n "${RAWVLA_EV_PHASES:-}" ]]; then
  for phase in $RAWVLA_EV_PHASES; do
    ev="${phase%%:*}"
    rep="${phase#*:}"
    if [[ -z "$ev" || -z "$rep" || "$ev" == "$rep" ]]; then
      echo "Invalid RAWVLA_EV_PHASES item: $phase" >&2
      exit 2
    fi
    run_phase "$ev" "$rep"
  done
else
  for ev in $EVS; do
    for rep in $REPS; do
      run_phase "$ev" "$rep"
    done
  done
fi

"$PY" "$ROOT/summarize_libero_float32_ev.py" "$RESULT_ROOT" || true
echo "[$(date -Is)] setting finished"
