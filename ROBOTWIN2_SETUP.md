# RoboTwin 2.0 setup

The pinned RoboTwin checkout lives at `third_party/RoboTwin` on the
`submodules/RoboTwin` branch. It includes RAW-camera and StarVLA policy
deployment integration.

## Install

From the RAW-VLA repository root:

```bash
git submodule update --init third_party/RoboTwin starVLA
conda env create -f environment-robotwin2.yml
conda activate robotwin
ROBOTWIN_PYTHON=python bash third_party/RoboTwin/script/_install.sh
```

The installer installs the pinned SAPIEN/MPLib stack, builds CuRobo 0.7.8, and
applies the compatibility patches expected by the RoboTwin checkout. Download
the benchmark assets using the instructions in `third_party/RoboTwin` before
running an episode.

## Validate

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  python scripts/check_environment.py robotwin --render
python scripts/replay_robotwin2_clean_training.py --help
python scripts/replay_robotwin2_paired_lighting_training.py --help
```

If the NVIDIA ICD lives elsewhere, set `VK_ICD_FILENAMES` to the path provided
by the host driver installation.

## Evaluate

```bash
export ROBOTWIN_CHECKPOINT=/absolute/path/to/checkpoint.pt
export ROBOTWIN_PYTHON="$(command -v python)"
scripts/robotwin2_eval.sh demo_clean all
scripts/robotwin2_eval.sh demo_randomized all
```

The runners derive repository paths from their own location. Override
`RAWVLA_ROOT`, `STARVLA_DIR`, or `ROBOTWIN_PATH` only when using checkouts
outside the standard repository layout.

See [ENVIRONMENTS.md](ENVIRONMENTS.md) for the complete dependency matrix and
the recorded local validation results.
