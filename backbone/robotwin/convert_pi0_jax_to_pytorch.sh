#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /path/to/pi0_robotwin/45000 /path/to/pi0_robotwin_pytorch_45000" >&2
  exit 2
fi

jax_checkpoint="$1"
output_checkpoint="$2"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
openpi_root="${OPENPI_ROOT:-$repository_root/third_party/RoboTwin/policy/pi05}"
openpi_python="${OPENPI_PYTHON:-$openpi_root/.venv/bin/python}"
converter="${OPENPI_CONVERTER:-$openpi_root/examples/convert_jax_model_to_pytorch.py}"

test -f "$jax_checkpoint/_CHECKPOINT_METADATA"
test -f "$jax_checkpoint/params/manifest.ocdbt"
test -f "$jax_checkpoint/assets/robotwin_50_clean/norm_stats.json"

if [[ ! -x "$openpi_python" ]]; then
  echo "OPENPI_PYTHON is not executable: $openpi_python" >&2
  exit 1
fi
if [[ ! -f "$converter" ]]; then
  echo "OpenPI converter not found: $converter" >&2
  echo "Use a compatible official OpenPI checkout and set OPENPI_ROOT or OPENPI_CONVERTER." >&2
  exit 1
fi
if [[ -e "$output_checkpoint" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
  echo "Output already exists: $output_checkpoint" >&2
  echo "Choose a new output path, or set ALLOW_EXISTING_OUTPUT=1 explicitly." >&2
  exit 1
fi

mkdir -p "$output_checkpoint"
(
  cd "$openpi_root"
  "$openpi_python" "$converter" \
    --checkpoint_dir "$jax_checkpoint" \
    --config_name pi0_base_aloha_robotwin_full \
    --output_path "$output_checkpoint" \
    --precision bfloat16
)

test -f "$output_checkpoint/model.safetensors"
test -f "$output_checkpoint/config.json"

"$openpi_python" - "$output_checkpoint/config.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
config = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "action_dim": 32,
    "action_horizon": 50,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"Unexpected {key} in {path}: {config.get(key)!r} != {value!r}")
print(f"Validated converted pi0 config: {path}")
PY

echo "Converted pi0 PyTorch checkpoint: $output_checkpoint"
echo "Keep using policy stats from: $jax_checkpoint/assets/robotwin_50_clean/norm_stats.json"
