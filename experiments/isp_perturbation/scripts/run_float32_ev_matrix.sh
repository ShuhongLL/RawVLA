#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RAWVLA_ROOT:-${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: run_float32_ev_matrix.sh

Runs the legacy synthetic float32 EV perturbation matrix. Override models,
suites, EV values, representations, GPU IDs, checkpoints, output paths, trial
count, and concurrency with environment variables documented in
experiments/isp_perturbation/README.md.
EOF
  exit 0
fi
REPO="${STARVLA_ROOT:-$ROOT/starVLA}"
PY="${STARVLA_PYTHON:-python}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/starVLA/playground/Pretrained_models}"
BASE="$CKPT_ROOT/base"
STAR="$CKPT_ROOT/StarVLA"
OPENPI="${OPENPI_CONVERTED_ROOT:-$ROOT/openpi_converted_protocol}"
OUT="${OUT:-$ROOT/results/isp-perturbation/libero-float32-ev}"
TRIALS="${NUM_TRIALS:-50}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
read -r -a GPU_LIST <<<"${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a SUITES <<<"${LIBERO_SUITES:-libero_spatial libero_object libero_goal libero_10}"
read -r -a MODELS <<<"${LIBERO_MODELS:-Qwen3-VL-OFT-LIBERO-4in1 Qwen3-VL-PI-LIBERO-4in1 WM4A-CosmoPredict-GR00T-LIBERO-4in1 WM4A-Wan2d2-OFT-LIBERO-4in1 PI0 PI05}"
read -r -a EVS <<<"${EV_VALUES:--8 -7 -6 3}"
read -r -a REPS <<<"${EV_REPRESENTATIONS:-raw_direct raw_recovered rgb_direct rgb_recovered}"

if (( ${#GPU_LIST[@]} == 0 || JOBS_PER_GPU < 1 )); then
  echo "GPU_IDS must not be empty and JOBS_PER_GPU must be positive" >&2
  exit 2
fi

mkdir -p "$OUT/control"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers"
export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_SKIP_VIDEO=1
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/.local/libero}"
export OPENPI_CONVERTED_ROOT="$OPENPI"
export PALIGEMMA_TOKENIZER="$BASE/paligemma_tokenizer.model"
unset DEBUG

wait_server() {
  local pid="$1" log="$2"
  for _ in $(seq 1 900); do
    kill -0 "$pid" 2>/dev/null || return 1
    grep -Eq 'server running|server listening' "$log" 2>/dev/null && return 0
    sleep 2
  done
  return 1
}

checkpoint_for() {
  case "$1" in
    Qwen3-VL-OFT-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_50000_pytorch_model.pt" ;;
    Qwen3-VL-PI-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_100000_pytorch_model.pt" ;;
    WM4A-CosmoPredict-GR00T-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_50000_pytorch_model.pt" ;;
    WM4A-Wan2d2-OFT-LIBERO-4in1) echo "$STAR/$1/checkpoints/steps_60000_pytorch_model.pt" ;;
    PI0) echo "$OPENPI/pi0_libero_starvla/bfloat16/model.safetensors" ;;
    PI05) echo "$OPENPI/pi05_libero_starvla/bfloat16/model.safetensors" ;;
  esac
}

server_overrides() {
  case "$1" in
    Qwen3-VL-PI-LIBERO-4in1)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$BASE/Qwen/Qwen3-VL-4B-Instruct" \
        --config_override framework.qwenvl.attn_implementation=sdpa \
        --config_override framework.action_model.diffusion_model_cfg.use_canonical_forward=false ;;
    Qwen3-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$BASE/Qwen/Qwen3-VL-4B-Instruct" \
        --config_override framework.qwenvl.attn_implementation=sdpa ;;
    WM4A-CosmoPredict-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$BASE/Qwen/Qwen3-VL-4B-Instruct" \
        --config_override framework.qwenvl.attn_implementation=sdpa \
        --config_override "framework.world_model.base_wm=$BASE/nvidia/Cosmos-Predict2-2B-Video2World" ;;
    WM4A-Wan2d2-*)
      printf '%s\n' --config_override "framework.qwenvl.base_vlm=$BASE/Qwen/Qwen3-VL-4B-Instruct" \
        --config_override framework.qwenvl.attn_implementation=sdpa \
        --config_override "framework.world_model.base_wm=$BASE/Wan-AI/Wan2.2-TI2V-5B-Diffusers" ;;
  esac
}

run_job() {
  local spec="$1" gpu="$2" worker="$3"
  local model suite ev rep
  IFS='|' read -r model suite ev rep <<<"$spec"
  local tag="${model}/${suite}/ev_${ev}/${rep}"
  local dir="$OUT/$tag" port=$((23000 + worker)) server_pid rc=0
  mkdir -p "$dir"
  [[ -f "$dir/job.done" ]] && return 0
  printf '[%s] START %s gpu=%s\n' "$(date -Is)" "$tag" "$gpu" | tee -a "$dir/job.log"
  export STARVLA_IMAGE_AUDIT_PATH="$dir/image_audit.jsonl"
  local ckpt
  ckpt="$(checkpoint_for "$model")"
  if [[ "$model" == PI0 || "$model" == PI05 ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/server_starvla_openpi.py \
      --model "$model" --model-source openpi --checkpoint "$ckpt" \
      --tokenizer "$PALIGEMMA_TOKENIZER" --precision bfloat16 --port "$port" --idle-timeout -1 \
      >>"$dir/server.log" 2>&1 &
  else
    mapfile -t overrides < <(server_overrides "$model")
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" deployment/model_server/server_policy.py \
      --ckpt_path "$ckpt" --port "$port" --use_bf16 --idle_timeout -1 \
      "${overrides[@]}" >>"$dir/server.log" 2>&1 &
  fi
  server_pid=$!
  echo "$server_pid" >"$dir/server.pid"
  if ! wait_server "$server_pid" "$dir/server.log"; then
    echo "server failed to become ready" >>"$dir/eval.log"
    rc=1
  elif [[ "$model" == PI0 || "$model" == PI05 ]]; then
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py \
      --model "$model" --model-source openpi --host 127.0.0.1 --port "$port" \
      --task-suite "$suite" --num-trials "$TRIALS" --max-tasks -1 \
      --result-json "$dir/result.json" --resume-path "$dir/episodes.jsonl" \
      --image-mode ev_float32 --exposure-ev "$ev" --ev-representation "$rep" \
      >>"$dir/eval.log" 2>&1
    rc=$?
    set -e
  else
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
      --args.pretrained-path "$ckpt" --args.host 127.0.0.1 --args.port "$port" \
      --args.task-suite-name "$suite" --args.num-trials-per-task "$TRIALS" --args.max-tasks -1 \
      --args.video-out-path "$dir/videos" --args.resume-path "$dir/episodes.jsonl" \
      --args.task-log-dir "$dir/tasks" --args.image-mode ev_float32 \
      --args.exposure-ev "$ev" --args.ev-representation "$rep" >>"$dir/eval.log" 2>&1
    rc=$?
    set -e
  fi
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  [[ "$rc" -eq 0 ]] && touch "$dir/job.done"
  printf '[%s] END %s rc=%s\n' "$(date -Is)" "$tag" "$rc" | tee -a "$dir/job.log"
}

jobs=()
for model in "${MODELS[@]}"; do
  for suite in "${SUITES[@]}"; do
    for ev in "${EVS[@]}"; do
      for rep in "${REPS[@]}"; do
        if [[ "$ev" == "-8" && "$rep" != "raw_recovered" ]]; then
          continue
        fi
        if [[ "$ev" == "-7" && ( "$rep" == "raw_direct" || "$rep" == "rgb_direct" ) ]]; then
          continue
        fi
        if [[ "$ev" == "-6" && ( "$rep" == "raw_direct" || "$rep" == "rgb_direct" ) ]]; then
          continue
        fi
        jobs+=("$model|$suite|$ev|$rep")
      done
    done
  done
done

worker_loop() {
  local worker="$1" worker_count="$2" idx gpu
  gpu="${GPU_LIST[$((worker % ${#GPU_LIST[@]}))]}"
  for ((idx=worker; idx<${#jobs[@]}; idx+=worker_count)); do
    run_job "${jobs[$idx]}" "$gpu" "$worker"
  done
}

echo "$$" >"$OUT/control/master.pid"
worker_count=$((${#GPU_LIST[@]} * JOBS_PER_GPU))
printf '[%s] scheduler start: %s jobs, %s GPUs, %s workers, %s trials/task\n' \
  "$(date -Is)" "${#jobs[@]}" "${#GPU_LIST[@]}" "$worker_count" "$TRIALS" | tee -a "$OUT/control/master.log"
"$PY" "$SCRIPT_DIR/summarize_float32_ev.py" "$OUT" --watch >"$OUT/control/summary.log" 2>&1 &
summary_pid=$!
trap 'kill "$summary_pid" 2>/dev/null || true' EXIT
pids=()
for worker in $(seq 0 $((worker_count - 1))); do
  worker_loop "$worker" "$worker_count" >>"$OUT/control/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
kill "$summary_pid" 2>/dev/null || true
"$PY" "$SCRIPT_DIR/summarize_float32_ev.py" "$OUT"
printf '[%s] scheduler finished\n' "$(date -Is)" | tee -a "$OUT/control/master.log"
