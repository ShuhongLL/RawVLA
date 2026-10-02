#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/pi05_robotwin" >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI 'hf'. Install huggingface_hub[cli]." >&2
  exit 1
fi

destination="$1"
repository="SidneyXie/pi05_robotwin"
revision="e49e2ab6c11f07511573b67261bd129e88d0a416"

mkdir -p "$destination"
hf download "$repository" \
  --revision "$revision" \
  --local-dir "$destination"

required_files=(
  config.json
  model.safetensors
  policy_preprocessor.json
  policy_preprocessor_step_3_normalizer_processor.safetensors
  policy_postprocessor.json
  policy_postprocessor_step_0_unnormalizer_processor.safetensors
  train_config.json
)
for relative_path in "${required_files[@]}"; do
  test -f "$destination/$relative_path"
done

echo "Downloaded pinned pi0.5 RoboTwin policy to: $destination"
echo "Run export_pi05_policy_norm_stats.py before RAWVLA training."
