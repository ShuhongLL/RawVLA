#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 /path/to/pi0_robotwin"
  exit 0
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/pi0_robotwin" >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI 'hf'. Install huggingface_hub[cli]." >&2
  exit 1
fi

destination="$1"
repository="Heisen0928/pi0_robotwin"
revision="f163ea4a7d4b4806a811644117e2e4e6ab29e1a9"

mkdir -p "$destination"
hf download "$repository" \
  --revision "$revision" \
  --include '45000/_CHECKPOINT_METADATA' '45000/params/**' '45000/assets/**' \
  --local-dir "$destination"

checkpoint="$destination/45000"
test -f "$checkpoint/_CHECKPOINT_METADATA"
test -f "$checkpoint/params/manifest.ocdbt"
test -f "$checkpoint/assets/robotwin_50_clean/norm_stats.json"

echo "Downloaded pinned pi0 RoboTwin checkpoint to: $checkpoint"
echo "Policy normalization stats: $checkpoint/assets/robotwin_50_clean/norm_stats.json"
