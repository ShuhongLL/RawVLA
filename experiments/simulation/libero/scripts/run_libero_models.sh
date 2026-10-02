#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RAWVLA_ROOT:-${ROOT:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: run_libero_models.sh

Environment-driven LIBERO model-matrix runner. Important variables:
  IMAGE_MODE                    rgb, raw_rgb10, ev_float32, or rawvla_bench
  RAW_FRONTEND                  none, identity, rawvla, darkisp, ram, raw_adapter, or rawild
  LIBERO_MODELS / LIBERO_SUITES space-separated model and suite lists
  GPU_IDS / JOBS_PER_GPU        worker placement
  NUM_TRIALS / MAX_TASKS        evaluation size
  RAW_FRONTEND_EVAL_DRY_RUN=1   print resolved checkpoints without launching
EOF
  exit 0
fi
REPO="${REPO:-$ROOT/starVLA}"
PY="${PY:-python}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-$ROOT/starVLA/playground/Pretrained_models}"
STAR="${STAR:-$PRETRAINED_ROOT/StarVLA}"
BASE="${BASE:-$PRETRAINED_ROOT}"
OPENPI="${OPENPI:-$ROOT/openpi_converted_protocol}"
QWEN25_BASE="${QWEN25_BASE:-$BASE/Qwen2.5-VL-3B-Instruct}"
QWEN25_FAST_BASE="${QWEN25_FAST_BASE:-$STAR/Qwen2.5-VL-3B-Instruct-Action}"
QWEN3_BASE="${QWEN3_BASE:-$BASE/Qwen3-VL-4B-Instruct}"
COSMOS_BASE="${COSMOS_BASE:-$BASE/nvidia/Cosmos-Predict2-2B-Video2World}"
WAN_BASE="${WAN_BASE:-$BASE/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
LOG_ROOT="${LOG_ROOT:-$ROOT/log/libero_zeroshot}"
TRIALS="${NUM_TRIALS:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
TASK_START="${TASK_START:-0}"
MAX_NEW_EPISODES_PER_RUN="${MAX_NEW_EPISODES_PER_RUN:--1}"
RUN_SET="${RUN_SET:-all}"
IMAGE_MODE="${IMAGE_MODE:-rgb}"
EV_REPRESENTATION="${EV_REPRESENTATION:-raw_direct}"
EXPOSURE_EV="${EXPOSURE_EV:-0.0}"
IMAGE_QUANTIZE_BITS="${IMAGE_QUANTIZE_BITS:-0}"
RAWVLA_BENCH_MANIFEST="${RAWVLA_BENCH_MANIFEST:-$ROOT/experiments/simulation/libero/manifests/libero_manifest_50_init_states_10000_rollouts.json}"
RAWVLA_BENCH_REPRESENTATION="${RAWVLA_BENCH_REPRESENTATION:-raw}"
USE_RAWVLA_MANIFEST_PLAN="${USE_RAWVLA_MANIFEST_PLAN:-0}"
RAW_FRONTEND="${RAW_FRONTEND:-none}"
RAW_FRONTEND_CHECKPOINT_ROOT="${RAW_FRONTEND_CHECKPOINT_ROOT:-$ROOT/baselines}"
RAW_FRONTEND_CHECKPOINT="${RAW_FRONTEND_CHECKPOINT:-}"
RAWVLA_CHECKPOINT_STEP="${RAWVLA_CHECKPOINT_STEP:-2000}"
RAWVLA_BURST_DENOISE_ETA="${RAWVLA_BURST_DENOISE_ETA:-0.5}"
RAWVLA_DISABLE_BURST_DENOISE="${RAWVLA_DISABLE_BURST_DENOISE:-false}"
RAWVLA_DISABLE_CHROMA_DESCRIPTOR="${RAWVLA_DISABLE_CHROMA_DESCRIPTOR:-false}"
RAWVLA_DISABLE_LUMA_DESCRIPTOR="${RAWVLA_DISABLE_LUMA_DESCRIPTOR:-false}"
RAWVLA_DISABLE_PREVIOUS_HIDDEN="${RAWVLA_DISABLE_PREVIOUS_HIDDEN:-false}"
RAWVLA_DISABLE_PREVIOUS_THETA="${RAWVLA_DISABLE_PREVIOUS_THETA:-false}"
RAWVLA_DISABLE_SPATIAL_FEATURE="${RAWVLA_DISABLE_SPATIAL_FEATURE:-false}"
RAWVLA_RECURRENT_MODE="${RAWVLA_RECURRENT_MODE:-split}"
WAN_VAE_POSTERIOR="${WAN_VAE_POSTERIOR:-}"
RAWVLA_FRONTEND_EXPORT_DIR="${RAWVLA_FRONTEND_EXPORT_DIR:-}"
RAWVLA_FRONTEND_EXPORT_SEED="${RAWVLA_FRONTEND_EXPORT_SEED:-20260823}"
RAW_FRONTEND_EVAL_DRY_RUN="${RAW_FRONTEND_EVAL_DRY_RUN:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-900}"
BASE_PORT="${BASE_PORT:-20000}"
read -r -a GPU_LIST <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a SUITES <<< "${LIBERO_SUITES:-libero_spatial libero_object libero_goal libero_10}"
read -r -a MODELS <<< "${LIBERO_MODELS:-Qwen3-VL-OFT-LIBERO-4in1 Qwen3-VL-PI-LIBERO-4in1 WM4A-CosmoPredict-GR00T-LIBERO-4in1 WM4A-Wan2d2-OFT-LIBERO-4in1 PI0 PI05}"

mkdir -p "$LOG_ROOT" "$LOG_ROOT/control"
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_SKIP_VIDEO=1
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/libero_config}"
export OPENPI_CONVERTED_ROOT="$OPENPI"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-$OPENPI/paligemma_tokenizer.model}"
unset DEBUG

if [[ "$IMAGE_MODE" != "rgb" && "$IMAGE_MODE" != "raw_rgb10" && "$IMAGE_MODE" != "ev_float32" && "$IMAGE_MODE" != "rawvla_bench" ]]; then
  echo "Unsupported IMAGE_MODE=$IMAGE_MODE (expected rgb, raw_rgb10, ev_float32, or rawvla_bench)" >&2
  exit 2
fi
if [[ ( "$IMAGE_MODE" == "rawvla_bench" || "$USE_RAWVLA_MANIFEST_PLAN" == "1" ) && ! -f "$RAWVLA_BENCH_MANIFEST" ]]; then
  echo "RAWVLA_BENCH_MANIFEST does not exist: $RAWVLA_BENCH_MANIFEST" >&2
  echo "Build it with: PYTHONPATH=$ROOT/experiments/simulation/libero python $ROOT/experiments/simulation/libero/scripts/build_manifest.py" >&2
  exit 2
fi
if [[ "$RAWVLA_BENCH_REPRESENTATION" != "raw" && "$RAWVLA_BENCH_REPRESENTATION" != "default_isp" ]]; then
  echo "RAWVLA_BENCH_REPRESENTATION must be raw or default_isp, got $RAWVLA_BENCH_REPRESENTATION" >&2
  exit 2
fi
case "$RAW_FRONTEND" in
  none|identity|rawvla|darkisp|ram|raw_adapter|rawild) ;;
  *)
    echo "RAW_FRONTEND must be one of none, identity, rawvla, darkisp, ram, raw_adapter, rawild; got $RAW_FRONTEND" >&2
    exit 2
    ;;
esac
if [[ "$RAW_FRONTEND" != "none" && "$IMAGE_MODE" != "rawvla_bench" ]]; then
  echo "RAW_FRONTEND=$RAW_FRONTEND requires IMAGE_MODE=rawvla_bench" >&2
  exit 2
fi
if [[ "$RAW_FRONTEND" != "none" && "$RAWVLA_BENCH_REPRESENTATION" != "raw" ]]; then
  echo "RAW_FRONTEND=$RAW_FRONTEND requires RAWVLA_BENCH_REPRESENTATION=raw" >&2
  exit 2
fi
if (( IMAGE_QUANTIZE_BITS < 0 || IMAGE_QUANTIZE_BITS > 8 )); then
  echo "IMAGE_QUANTIZE_BITS must be in [0, 8], got $IMAGE_QUANTIZE_BITS" >&2
  exit 2
fi
if (( JOBS_PER_GPU < 1 || JOBS_PER_GPU > 3 )); then
  echo "JOBS_PER_GPU must be between 1 and 3, got $JOBS_PER_GPU" >&2
  exit 2
fi
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU index" >&2
  exit 2
fi
for gpu_id in "${GPU_LIST[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_IDS contains an invalid GPU index/group: $gpu_id" >&2
    exit 2
  fi
done
if [[ -n "$WAN_VAE_POSTERIOR" && "$WAN_VAE_POSTERIOR" != mode && "$WAN_VAE_POSTERIOR" != sample ]]; then
  echo "WAN_VAE_POSTERIOR must be mode, sample, or empty; got $WAN_VAE_POSTERIOR" >&2
  exit 2
fi

checkpoint_for() {
  case "$1" in
    Qwen2.5-VL-FAST-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_30000_pytorch_model.pt" ;;
    Qwen2.5-VL-GR00T-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_30000_pytorch_model.pt" ;;
    Qwen2.5-VL-OFT-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_30000_pytorch_model.pt" ;;
    Qwen3-VL-OFT-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_50000_pytorch_model.pt" ;;
    Qwen3-VL-PI-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_100000_pytorch_model.pt" ;;
    WM4A-CosmoPredict-GR00T-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_50000_pytorch_model.pt" ;;
    WM4A-Wan2d2-OFT-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_60000_pytorch_model.pt" ;;
    PI0) echo "$OPENPI/pi0_libero_starvla/bfloat16/model.safetensors" ;;
    PI05) echo "$OPENPI/pi05_libero_starvla/bfloat16/model.safetensors" ;;
    *) return 1 ;;
  esac
}

backbone_key() {
  case "$1" in
    Qwen3-VL-OFT-LIBERO-4in1) echo qwen3_oft ;;
    Qwen3-VL-PI-LIBERO-4in1) echo qwen3_pi ;;
    WM4A-CosmoPredict-GR00T-LIBERO-4in1) echo wm4a_cosmos ;;
    WM4A-Wan2d2-OFT-LIBERO-4in1) echo wm4a_wan ;;
    PI0) echo pi0 ;;
    PI05) echo pi05 ;;
    *) return 1 ;;
  esac
}

frontend_checkpoint_for() {
  local model="$1" key step_k
  key="$(backbone_key "$model")"
  case "$RAW_FRONTEND" in
    rawvla)
      if [[ -n "$RAW_FRONTEND_CHECKPOINT" ]]; then
        echo "$RAW_FRONTEND_CHECKPOINT"
        return 0
      fi
      if (( RAWVLA_CHECKPOINT_STEP < 1000 || RAWVLA_CHECKPOINT_STEP % 1000 != 0 )); then
        echo "RAWVLA_CHECKPOINT_STEP must be a positive multiple of 1000; got $RAWVLA_CHECKPOINT_STEP" >&2
        return 1
      fi
      step_k="$((RAWVLA_CHECKPOINT_STEP / 1000))k"
      echo "$RAW_FRONTEND_CHECKPOINT_ROOT/rawvla/checkpoints/libero/$key/$step_k/steps_${RAWVLA_CHECKPOINT_STEP}_raw_frontend_pytorch_model.pt"
      ;;
    darkisp) echo "$RAW_FRONTEND_CHECKPOINT_ROOT/darkisp/checkpoints/libero/$key/darkisp_step_0008000.pt" ;;
    ram) echo "$RAW_FRONTEND_CHECKPOINT_ROOT/ram/checkpoints/libero/$key/ram_step_0008000.pt" ;;
    raw_adapter)
      if [[ "$key" == pi05 ]]; then
        echo "$RAW_FRONTEND_CHECKPOINT_ROOT/raw_adapter/checkpoints/libero/$key/raw_adapter_step_0004000.pt"
      else
        echo "$RAW_FRONTEND_CHECKPOINT_ROOT/raw_adapter/checkpoints/libero/$key/raw_adapter_step_0008000.pt"
      fi
      ;;
    rawild) echo "$RAW_FRONTEND_CHECKPOINT_ROOT/rawild/checkpoints/libero/$key/rawild_step_0008000.pt" ;;
    none|identity) return 0 ;;
  esac
}

frontend_overrides() {
  local model="$1" override_flag="${2:---config_override}" frontend_checkpoint
  if [[ "$RAW_FRONTEND" == none ]]; then
    return 0
  fi
  printf '%s\n' "$override_flag" "framework.raw_frontend.name=$RAW_FRONTEND"
  if [[ "$RAW_FRONTEND" == rawvla ]]; then
    printf '%s\n' "$override_flag" framework.raw_frontend.source_key=raw_burst_float32
  else
    printf '%s\n' "$override_flag" framework.raw_frontend.source_key=image
  fi
  printf '%s\n' "$override_flag" framework.raw_frontend.input_format=rgb_raw
  printf '%s\n' "$override_flag" framework.raw_frontend.trainable=false
  if [[ "$RAW_FRONTEND" != identity ]]; then
    frontend_checkpoint="$(frontend_checkpoint_for "$model")"
    if [[ ! -f "$frontend_checkpoint" ]]; then
      echo "Frontend checkpoint does not exist: $frontend_checkpoint" >&2
      return 1
    fi
    printf '%s\n' "$override_flag" "framework.raw_frontend.checkpoint=$frontend_checkpoint"
  fi
  if [[ "$RAW_FRONTEND" == rawvla ]]; then
    printf '%s\n' "$override_flag" framework.raw_frontend.burst_frames=6
    printf '%s\n' "$override_flag" framework.raw_frontend.rnn_frames=6
    printf '%s\n' "$override_flag" framework.raw_frontend.state_dim=128
    printf '%s\n' "$override_flag" framework.raw_frontend.luma_state_dim=64
    printf '%s\n' "$override_flag" framework.raw_frontend.split_luma_chroma_condition=true
    printf '%s\n' "$override_flag" framework.raw_frontend.max_exposure_ev=6.0
    printf '%s\n' "$override_flag" "framework.raw_frontend.burst_denoise_eta=$RAWVLA_BURST_DENOISE_ETA"
    printf '%s\n' "$override_flag" framework.raw_frontend.fixed_update_alpha=1.0
    printf '%s\n' "$override_flag" "framework.raw_frontend.disable_burst_denoise=$RAWVLA_DISABLE_BURST_DENOISE"
    printf '%s\n' "$override_flag" "framework.raw_frontend.disable_chroma_descriptor=$RAWVLA_DISABLE_CHROMA_DESCRIPTOR"
    printf '%s\n' "$override_flag" "framework.raw_frontend.disable_luma_descriptor=$RAWVLA_DISABLE_LUMA_DESCRIPTOR"
    printf '%s\n' "$override_flag" "framework.raw_frontend.disable_previous_hidden=$RAWVLA_DISABLE_PREVIOUS_HIDDEN"
    printf '%s\n' "$override_flag" "framework.raw_frontend.disable_previous_theta=$RAWVLA_DISABLE_PREVIOUS_THETA"
    printf '%s\n' "$override_flag" "framework.raw_frontend.disable_spatial_feature=$RAWVLA_DISABLE_SPATIAL_FEATURE"
    printf '%s\n' "$override_flag" "framework.raw_frontend.recurrent_mode=$RAWVLA_RECURRENT_MODE"
  fi
}

if [[ "$RAW_FRONTEND" != none && "$RAW_FRONTEND" != identity ]]; then
  for model in "${MODELS[@]}"; do
    frontend_checkpoint="$(frontend_checkpoint_for "$model")"
    if [[ ! -f "$frontend_checkpoint" ]]; then
      echo "Frontend checkpoint does not exist: $frontend_checkpoint" >&2
      exit 2
    fi
  done
fi

if [[ "$RAW_FRONTEND_EVAL_DRY_RUN" == 1 ]]; then
  echo "frontend=$RAW_FRONTEND image_mode=$IMAGE_MODE raw_representation=$RAWVLA_BENCH_REPRESENTATION"
  for model in "${MODELS[@]}"; do
    backbone_checkpoint="$(checkpoint_for "$model")"
    frontend_checkpoint="$(frontend_checkpoint_for "$model")"
    echo "model=$model backbone=$backbone_checkpoint frontend_checkpoint=${frontend_checkpoint:-<none>}"
  done
  exit 0
fi

server_overrides() {
  case "$1" in
    Qwen2.5-VL-FAST-LIBERO-4in1)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN25_FAST_BASE" --config_override framework.qwenvl.attn_implementation=sdpa ;;
    Qwen2.5-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN25_BASE" --config_override framework.qwenvl.attn_implementation=sdpa ;;
    WM4A-CosmoPredict-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN3_BASE" --config_override framework.qwenvl.attn_implementation=sdpa --config_override "framework.world_model.base_wm=$COSMOS_BASE" ;;
    WM4A-Wan2d2-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN3_BASE" --config_override framework.qwenvl.attn_implementation=sdpa --config_override "framework.world_model.base_wm=$WAN_BASE"
      if [[ -n "$WAN_VAE_POSTERIOR" ]]; then
        printf '%s\n' --config_override "framework.world_model.vae_posterior=$WAN_VAE_POSTERIOR"
      fi
      ;;
    Qwen3-VL-PI-LIBERO-4in1)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN3_BASE" --config_override framework.qwenvl.attn_implementation=sdpa --config_override framework.action_model.diffusion_model_cfg.use_canonical_forward=false ;;
    Qwen3-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$QWEN3_BASE" --config_override framework.qwenvl.attn_implementation=sdpa ;;
  esac
}

wait_server() {
  local pid="$1" log="$2"
  for _ in $(seq 1 "$SERVER_WAIT_ATTEMPTS"); do
    kill -0 "$pid" 2>/dev/null || return 1
    grep -Eq 'server running|server listening' "$log" 2>/dev/null && return 0
    sleep 2
  done
  return 1
}

run_starvla_job() {
  local model="$1" suite="$2" gpu="$3" port="$4" dir="$5"
  local checkpoint server_pid rc
  local -a overrides=() frontend_args=()
  checkpoint="$(checkpoint_for "$model")"
  mapfile -t overrides < <(server_overrides "$model")
  if [[ "$RAW_FRONTEND" != none ]]; then
    mapfile -t frontend_args < <(frontend_overrides "$model" --config_override)
  fi
  local rawvla_args=()
  if [[ "$IMAGE_MODE" == "rawvla_bench" || "$USE_RAWVLA_MANIFEST_PLAN" == "1" ]]; then
    rawvla_args=(--args.rawvla-bench-manifest "$RAWVLA_BENCH_MANIFEST" --args.rawvla-bench-representation "$RAWVLA_BENCH_REPRESENTATION")
  fi
  if [[ -n "$RAWVLA_FRONTEND_EXPORT_DIR" ]]; then
    rawvla_args+=(--args.rawvla-frontend-export-dir "$RAWVLA_FRONTEND_EXPORT_DIR" --args.rawvla-frontend-export-seed "$RAWVLA_FRONTEND_EXPORT_SEED")
  fi
  if [[ "$RAW_FRONTEND" == none ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" deployment/model_server/server_policy.py \
      --ckpt_path "$checkpoint" --port "$port" --use_bf16 --idle_timeout -1 \
      "${overrides[@]}" >>"$dir/server.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" deployment/model_server/server_policy.py \
      --ckpt_path "$checkpoint" --port "$port" --use_bf16 --idle_timeout -1 \
      "${overrides[@]}" "${frontend_args[@]}" >>"$dir/server.log" 2>&1 &
  fi
  server_pid=$!
  echo "$server_pid" >"$dir/server.pid"
  if ! wait_server "$server_pid" "$dir/server.log"; then
    echo "server failed to become ready" >>"$dir/eval.log"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    return 1
  fi
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py     --args.pretrained-path "$checkpoint" --args.host 127.0.0.1 --args.port "$port"     --args.task-suite-name "$suite" --args.num-trials-per-task "$TRIALS" --args.task-start "$TASK_START" --args.max-tasks "$MAX_TASKS"     --args.max-new-episodes-per-run "$MAX_NEW_EPISODES_PER_RUN" --args.video-out-path "$dir/videos" --args.resume-path "$dir/episodes.jsonl"     --args.task-log-dir "$dir/tasks" --args.job-name "${model}_${suite}" --args.image-mode "$IMAGE_MODE"     --args.ev-representation "$EV_REPRESENTATION" --args.exposure-ev "$EXPOSURE_EV"     --args.image-quantize-bits "$IMAGE_QUANTIZE_BITS" "${rawvla_args[@]}" >>"$dir/eval.log" 2>&1
  rc=$?
  set -e
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  return "$rc"
}

run_openpi_job() {
  local model="$1" suite="$2" gpu="$3" port="$4" dir="$5"
  local checkpoint server_pid rc=0 task
  local -a frontend_args=()
  checkpoint="$(checkpoint_for "$model")"
  if [[ "$RAW_FRONTEND" != none ]]; then
    mapfile -t frontend_args < <(frontend_overrides "$model" --config-override)
  fi
  local rawvla_args=()
  if [[ "$IMAGE_MODE" == "rawvla_bench" || "$USE_RAWVLA_MANIFEST_PLAN" == "1" ]]; then
    rawvla_args=(--rawvla-bench-manifest "$RAWVLA_BENCH_MANIFEST" --rawvla-bench-representation "$RAWVLA_BENCH_REPRESENTATION")
  fi
  if [[ "$RAW_FRONTEND" == none ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/server_starvla_openpi.py \
      --model "$model" --model-source openpi --checkpoint "$checkpoint" \
      --tokenizer "$PALIGEMMA_TOKENIZER" --precision bfloat16 --port "$port" --idle-timeout -1 \
      >>"$dir/server.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/server_starvla_openpi.py \
      --model "$model" --model-source openpi --checkpoint "$checkpoint" \
      --tokenizer "$PALIGEMMA_TOKENIZER" --precision bfloat16 --port "$port" --idle-timeout -1 \
      "${frontend_args[@]}" >>"$dir/server.log" 2>&1 &
  fi
  server_pid=$!
  echo "$server_pid" >"$dir/server.pid"
  if ! wait_server "$server_pid" "$dir/server.log"; then
    echo "server failed to become ready" >>"$dir/eval.log"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    return 1
  fi
  local task_last=9
  if (( MAX_TASKS > 0 )); then task_last=$((TASK_START + MAX_TASKS - 1)); fi
  for task in $(seq "$TASK_START" "$task_last"); do
    [[ -f "$dir/task_${task}.done" ]] && continue
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py       --model "$model" --model-source openpi --host 127.0.0.1 --port "$port"       --task-suite "$suite" --num-trials "$TRIALS" --task-start "$task" --max-tasks 1       --result-json "$dir/task_${task}.json" --resume-path "$dir/task_${task}.episodes.jsonl"       --image-mode "$IMAGE_MODE" --ev-representation "$EV_REPRESENTATION"       --exposure-ev "$EXPOSURE_EV" --image-quantize-bits "$IMAGE_QUANTIZE_BITS" "${rawvla_args[@]}" >>"$dir/task_${task}.log" 2>&1
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
      touch "$dir/task_${task}.done"
    else
      echo "task $task failed with rc=$rc" >>"$dir/eval.log"
      break
    fi
  done
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  return "$rc"
}

run_job() {
  local model="$1" suite="$2" gpu="$3" worker="$4"
  local port=$((BASE_PORT + worker)) dir="$LOG_ROOT/$RAW_FRONTEND/$model/$suite" rc=0
  mkdir -p "$dir"
  if [[ -f "$dir/job.done" ]]; then
    echo "[$(date -Is)] skip completed $model $suite"
    return 0
  fi
  printf '[%s] START model=%s frontend=%s suite=%s gpu=%s worker=%s port=%s trials=%s image_mode=%s raw_rep=%s ev_rep=%s bits=%s\n'     "$(date -Is)" "$model" "$RAW_FRONTEND" "$suite" "$gpu" "$worker" "$port" "$TRIALS" "$IMAGE_MODE" "$RAWVLA_BENCH_REPRESENTATION" "$EV_REPRESENTATION" "$IMAGE_QUANTIZE_BITS" | tee -a "$dir/job.log"
  if [[ "$model" == PI0 || "$model" == PI05 ]]; then
    run_openpi_job "$model" "$suite" "$gpu" "$port" "$dir" || rc=$?
  else
    run_starvla_job "$model" "$suite" "$gpu" "$port" "$dir" || rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then touch "$dir/job.done"; fi
  printf '[%s] END model=%s suite=%s gpu=%s worker=%s rc=%s\n'     "$(date -Is)" "$model" "$suite" "$gpu" "$worker" "$rc" | tee -a "$dir/job.log"
  return "$rc"
}

jobs=()
if [[ "$RUN_SET" == "repair" ]]; then
  for suite in "${SUITES[@]}"; do jobs+=("Qwen3-VL-PI-LIBERO-4in1|$suite"); done
  jobs+=("WM4A-Wan2d2-OFT-LIBERO-4in1|libero_goal")
  for model in PI0 PI05; do
    for suite in "${SUITES[@]}"; do jobs+=("$model|$suite"); done
  done
else
  for model in "${MODELS[@]}"; do
    for suite in "${SUITES[@]}"; do jobs+=("$model|$suite"); done
  done
fi

worker_loop() {
  local worker="$1" worker_count="$2" gpu idx model suite
  gpu="${GPU_LIST[$(((worker / JOBS_PER_GPU) % ${#GPU_LIST[@]}))]}"
  for ((idx=worker; idx<${#jobs[@]}; idx+=worker_count)); do
    IFS='|' read -r model suite <<<"${jobs[$idx]}"
    run_job "$model" "$suite" "$gpu" "$worker"
  done
}

echo "$$" >"$LOG_ROOT/control/master.pid"
worker_count=$((${#GPU_LIST[@]} * JOBS_PER_GPU))
if (( ${#jobs[@]} < worker_count )); then worker_count=${#jobs[@]}; fi
printf '[%s] scheduler start: %s jobs, %s workers, run_set=%s, image_mode=%s, ev_rep=%s, bits=%s, jobs_per_gpu=%s, %s trials/task\n'   "$(date -Is)" "${#jobs[@]}" "$worker_count" "$RUN_SET" "$IMAGE_MODE" "$EV_REPRESENTATION" "$IMAGE_QUANTIZE_BITS" "$JOBS_PER_GPU" "$TRIALS" | tee -a "$LOG_ROOT/control/master.log"
pids=()
for worker in $(seq 0 $((worker_count - 1))); do
  worker_loop "$worker" "$worker_count" >>"$LOG_ROOT/control/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
"$PY" "$SCRIPT_DIR/summarize_libero.py" "$LOG_ROOT" >"$LOG_ROOT/summary.txt"
printf '[%s] scheduler finished\n' "$(date -Is)" | tee -a "$LOG_ROOT/control/master.log"
