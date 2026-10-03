#!/usr/bin/env bash
set -euo pipefail

# Canonical RoboTwin2 VLA closed-loop replay/rollout entry. Actions come from
# StarVLA model.step(), never from expert left_joint_path/right_joint_path.
export MODE="${MODE:-easy}"
export MATCH_DATASET_SEEDS="${MATCH_DATASET_SEEDS:-1}"
export FORCE_EVAL_SEED="${FORCE_EVAL_SEED:-1}"
export RECORD_EPISODE_DATA="${RECORD_EPISODE_DATA:-1}"
export ORIGINAL_IMAGE_KEY="${ORIGINAL_IMAGE_KEY:-rgb}"
export POLICY_IMAGE_KEY="${POLICY_IMAGE_KEY:-default_isp}"

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec bash "${CODE_ROOT}/scripts/evaluation/run_robotwin2_policy_easy.sh"
