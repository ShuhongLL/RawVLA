#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd -- "$BENCH_DIR/../.." && pwd)"

MANIFEST="${RAWVLA_BENCH_MANIFEST:-$BENCH_DIR/libero_manifest_50_init_states_10000_rollouts.json}"
REPRESENTATION="${RAWVLA_BENCH_REPRESENTATION:-raw}"

if [[ ! -f "$MANIFEST" ]]; then
  "${PYTHON_BIN:-${PY:-python}}" "$SCRIPT_DIR/build_manifest.py" --out "$MANIFEST" --base-seeds-per-task 50
fi

if [[ "$REPRESENTATION" != "raw" && "$REPRESENTATION" != "default_isp" ]]; then
  echo "RAWVLA_BENCH_REPRESENTATION must be raw or default_isp, got $REPRESENTATION" >&2
  exit 2
fi

PY="${PY:-python}" \
ROOT="${ROOT:-$ROOT}" \
IMAGE_MODE=rawvla_bench \
RAWVLA_BENCH_MANIFEST="$MANIFEST" \
RAWVLA_BENCH_REPRESENTATION="$REPRESENTATION" \
LOG_ROOT="${LOG_ROOT:-$ROOT/results/rawvla-bench-light-libero/$REPRESENTATION}" \
bash "$ROOT/run_libero_zeroshot_all.sh"
