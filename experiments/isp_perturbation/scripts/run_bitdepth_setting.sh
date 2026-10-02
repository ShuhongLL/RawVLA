#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_bitdepth_setting.sh MODEL SUITE BITS [TRIALS]

Runs one bit-depth setting on one machine:
  1. ev_float32/raw_direct      (unprocess -> raw float32)
  2. ev_float32/rgb_recovered   (unprocess -> reprocess RGB float32)

Each phase has independent logs, resume ledger, and result JSON.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
  exit 2
fi

MODEL="$1"
SUITE="$2"
BITS="$3"
TRIALS="${4:-${NUM_TRIALS:-50}}"
MAX_TASKS="${MAX_TASKS:--1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RAWVLA_ROOT:-${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
DATA_ROOT="${DATA_ROOT:-$ROOT}"
REPO="${REPO:-$ROOT/starVLA}"
RESULT_ROOT="${RESULT_ROOT:-$DATA_ROOT/results/libero-bitdepth-zeroshot}"
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
EXPOSURE_EV="${EXPOSURE_EV:-0.0}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
SEED="${SEED:-7}"

if [[ -z "${PY:-}" && -f "$RAWVLA_ACTIVATE" ]]; then
  # The Volc runtime image can be older than the packed conda env; this helper
  # selects a compatible Python path and overlays the missing Python packages.
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
    "$HOME/miniforge3/envs/starVLA/bin/python"
    "$HOME/miniforge3/envs/starvla/bin/python"
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
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  echo "Could not find a Python executable; set PY or STARVLA_PYTHON." >&2
  return 1
}

PY="$(resolve_python)"
echo "[$(date -Is)] python=$PY"

case "$SUITE" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *) echo "Unsupported suite: $SUITE" >&2; exit 2 ;;
esac
case "$BITS" in
  2|3|4|5|6) ;;
  *) echo "BITS must be one of 6, 5, 4, 3, 2; got $BITS" >&2; exit 2 ;;
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
SETTING_DIR="$RESULT_ROOT/$MODEL_SAFE/$SUITE/bits_${BITS}"
CONTROL_DIR="$RESULT_ROOT/control"
SERVER_LOG="$SETTING_DIR/server.log"
JOB_LOG="$SETTING_DIR/job.log"
mkdir -p "$SETTING_DIR" "$CONTROL_DIR"

exec > >(tee -a "$JOB_LOG") 2>&1

echo "[$(date -Is)] setting start model=$MODEL suite=$SUITE bits=$BITS trials=$TRIALS max_tasks=$MAX_TASKS gpu=$GPU port=$PORT"
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

LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$CONTROL_DIR/.libero}"
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

if [[ -f "$SETTING_DIR/raw/job.done" && -f "$SETTING_DIR/rgb/job.done" ]]; then
  echo "[$(date -Is)] setting already complete; refreshing summary only"
  "$PY" "$SCRIPT_DIR/summarize_runs.py" "$RESULT_ROOT" >"$RESULT_ROOT/summary.txt"
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
    rm -f "$SETTING_DIR/server.pid"
    server_pid=""
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  if [[ -f "$SETTING_DIR/server.pid" ]]; then
    kill_server_tree "$(cat "$SETTING_DIR/server.pid" 2>/dev/null || true)"
    rm -f "$SETTING_DIR/server.pid"
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
  echo "$server_pid" >"$SETTING_DIR/server.pid"
  wait_server "$server_pid" "$SERVER_LOG"
}

run_phase() {
  local label="$1"
  local representation="$2"
  local phase_dir="$SETTING_DIR/$label"
  local eval_log="$phase_dir/eval.log"
  local result_json="$phase_dir/result.json"
  local resume="$phase_dir/episodes.jsonl"
  mkdir -p "$phase_dir/tasks" "$phase_dir/videos"
  if [[ -f "$phase_dir/job.done" ]]; then
    echo "[$(date -Is)] skip completed phase=$label"
    return 0
  fi
  echo "[$(date -Is)] phase start label=$label representation=$representation bits=$BITS"
  set +e
  if [[ "$MODEL" == "PI0" || "$MODEL" == "PI05" || "$MODEL" == "PI0.5" ]]; then
    local pi_model="$MODEL"
    [[ "$pi_model" == "PI0.5" ]] && pi_model="PI05"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py \
      --model "$pi_model" --model-source openpi --host 127.0.0.1 --port "$PORT" \
      --task-suite "$SUITE" --num-trials "$TRIALS" --max-tasks "$MAX_TASKS" --resize-size "$RESIZE_SIZE" \
      --seed "$SEED" --result-json "$result_json" --resume-path "$resume" \
      --image-mode ev_float32 --ev-representation "$representation" --exposure-ev "$EXPOSURE_EV" \
      --image-quantize-bits "$BITS" 2>&1 | tee -a "$eval_log"
  else
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
      --args.pretrained-path "$CKPT" --args.host 127.0.0.1 --args.port "$PORT" \
      --args.task-suite-name "$SUITE" --args.num-trials-per-task "$TRIALS" --args.max-tasks "$MAX_TASKS" \
      --args.video-out-path "$phase_dir/videos" --args.resume-path "$resume" \
      --args.task-log-dir "$phase_dir/tasks" --args.job-name "${MODEL}_${SUITE}_${label}_bits${BITS}" \
      --args.image-mode ev_float32 --args.ev-representation "$representation" \
      --args.exposure-ev "$EXPOSURE_EV" --args.image-quantize-bits "$BITS" 2>&1 | tee -a "$eval_log"
  fi
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -eq 0 ]]; then
    touch "$phase_dir/job.done"
    echo "[$(date -Is)] phase done label=$label"
  else
    echo "[$(date -Is)] phase failed label=$label rc=$rc"
  fi
  return "$rc"
}

if ! start_server; then
  echo "[$(date -Is)] server failed to become ready; see $SERVER_LOG" >&2
  exit 1
fi
echo "[$(date -Is)] server ready"

run_phase raw raw_direct
run_phase rgb rgb_recovered

"$PY" "$SCRIPT_DIR/summarize_runs.py" "$RESULT_ROOT" >"$RESULT_ROOT/summary.txt"
echo "[$(date -Is)] setting finished"
