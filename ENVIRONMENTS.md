# Environment installation and validation

RAW-VLA uses separate environments because the VLA backbones and simulators
pin incompatible PyTorch, CUDA, TensorFlow, and simulator stacks. Run every
command below from the repository root unless it explicitly changes directory.

## Common preparation

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Install an NVIDIA driver compatible with the selected PyTorch wheel. The
checked-in Conda files describe Python dependencies; system NVIDIA, Vulkan/EGL,
compiler, and simulator asset installation remains host-specific.

## StarVLA + LIBERO + RAW-VLA training and replay

This is the main RAW-VLA environment. The checked-in pins match the environment
used for the local smoke tests: Python 3.10, PyTorch 2.6/CUDA 12.4,
Transformers 4.57, MuJoCo 3.2.3, and robosuite 1.4.0.

```bash
conda env create -f environment-starvla-libero.yml
conda activate starvla-libero
python -m pip install -e ./LIBERO-git
python -m pip install --no-deps -e ./third_party/openvla-oft
python -m pip install --no-deps -e ./starVLA
python scripts/configure_libero.py
```

`--no-deps` on the two VLA source installs preserves the tested dependency pins
from the environment file instead of allowing their upstream metadata to
downgrade PyTorch. The generated `libero_config/config.yaml` is machine-local
and ignored by Git. Put demonstrations under the default `benchmark_data/`
tree or pass `--datasets` to the configuration command.

Validate imports, RAW-VLA computation, and optional headless LIBERO rendering:

```bash
python scripts/check_environment.py rawvla
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python scripts/check_environment.py starvla-libero --render
python scripts/replay_libero_train_rawvla_light_npz.py --help
```

## RoboTwin 2.0

```bash
conda env create -f environment-robotwin2.yml
conda activate robotwin
ROBOTWIN_PYTHON=python bash third_party/RoboTwin/script/_install.sh
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  python scripts/check_environment.py robotwin --render
python scripts/replay_robotwin2_clean_training.py --help
python scripts/replay_robotwin2_paired_lighting_training.py --help
```

The RoboTwin installer builds CuRobo and applies the SAPIEN/MPLib compatibility
patches required by the pinned RoboTwin checkout. Download RoboTwin assets as
described by its submodule before running an episode.

## FastWAM

FastWAM uses a newer PyTorch/CUDA stack and should not share the StarVLA env.

```bash
conda env create -f environment-fastwam.yml
conda activate fastwam
python -m pip install -e ./third_party/FastWAM
python scripts/check_environment.py fastwam
```

The pinned environment uses Python 3.10, PyTorch 2.7.1/CUDA 12.8, and
torchvision 0.22.1, matching the locally tested FastWAM installation.

## OpenVLA-OFT standalone

Use this only for standalone OpenVLA-OFT training/evaluation. StarVLA integration
uses the main StarVLA environment above.

```bash
conda env create -f environment-openvla-oft.yml
conda activate openvla-oft
python -m pip install -e ./third_party/openvla-oft
python -m pip install "flash-attn==2.5.5" --no-build-isolation
python scripts/check_environment.py openvla-oft
```

## Isaac GR00T

The GR00T submodule owns an `uv.lock`, so its locked `uv` environment is the
authority. The top-level Conda file is only a Python/system-tool bootstrap.

```bash
conda env create -f environment-groot.yml
conda activate groot
cd third_party/Isaac-GR00T
uv sync --frozen
uv run python ../../scripts/check_environment.py groot
```

For architecture-specific NVIDIA platforms, follow the deployment instructions
in the pinned GR00T checkout instead of mixing those dependencies into another
RAW-VLA environment.

## Exact snapshots

`benchmark/docs/env_repro/` contains package snapshots from the tested StarVLA
and RoboTwin environments. They are auditing aids, not a replacement for the
short, reviewed environment files above: exact exports can include platform
build identifiers that do not solve on another machine.
