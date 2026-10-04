# RoboTwin 2.0 setup

The pinned RoboTwin checkout lives at `third_party/RoboTwin` on the
`submodules/RoboTwin` branch. It includes RAW-camera and StarVLA policy
deployment integration.

RAW-VLA π0/π0.5 training uses the separate StarVLA environment and the
published paired trajectories. See [`configs/robotwin`](../../../configs/robotwin/README.md)
for the training recipes; this page installs the RoboTwin simulator for evaluation.

## Install

From the RAW-VLA repository root:

```bash
git submodule update --init third_party/RoboTwin starVLA
conda env create -f environment-robotwin2.yml
conda activate robotwin
# Download the official assets, or set ROBOTWIN_ASSETS_SOURCE to reuse them.
bash scripts/setup/setup_robotwin_assets.sh
ROBOTWIN_PYTHON=python bash third_party/RoboTwin/script/_install.sh
```

The installer installs the pinned SAPIEN/MPLib stack, builds CuRobo 0.7.8, and
applies the compatibility patches expected by the RoboTwin checkout. The asset
setup script downloads and extracts the roughly 30 GB simulator asset set. To
reuse an existing download instead, run
`ROBOTWIN_ASSETS_SOURCE=/path/to/RoboTwin/assets bash scripts/setup/setup_robotwin_assets.sh`.

## Validate

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  python scripts/setup/check_environment.py robotwin --render
```

If the NVIDIA ICD lives elsewhere, set `VK_ICD_FILENAMES` to the path provided
by the host driver installation.

## Evaluate

```bash
export ROBOTWIN_CHECKPOINT=/absolute/path/to/checkpoint.pt
export ROBOTWIN_PYTHON="$(command -v python)"
scripts/evaluation/robotwin2_eval.sh demo_clean all
scripts/evaluation/robotwin2_eval.sh demo_randomized all
```

The runners derive repository paths from their own location. Override
`RAWVLA_ROOT`, `STARVLA_DIR`, or `ROBOTWIN_PATH` only when using checkouts
outside the standard repository layout.

See [ENVIRONMENTS.md](../../../ENVIRONMENTS.md) for the complete dependency matrix and
the recorded local validation results.
