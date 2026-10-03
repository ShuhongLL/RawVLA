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

This is the main RAW-VLA environment. The checked-in pins match the validated
environment: Python 3.10, PyTorch 2.6/CUDA 12.4,
Transformers 4.57, MuJoCo 3.2.3, and robosuite 1.4.0.

```bash
conda env create -f environment-libero.yml
conda activate starvla-libero
python -m pip install --no-deps --no-build-isolation -e ./third_party/LIBERO
python -m pip install --no-deps -e ./third_party/openvla-oft
python -m pip install --no-deps -e ./starVLA
export LIBERO_CONFIG_PATH="$PWD/.local/libero"
python scripts/setup/configure_libero.py
```

`--no-deps` on the source installs preserves the tested dependency pins
from the environment file. The LIBERO editable install uses the environment's
setuptools without downloading a separate build environment. The generated
`.local/libero/config.yaml` is machine-local and ignored by Git. Put
demonstrations under the default `benchmark_data/`
tree or pass `--datasets` to the configuration command.
If LIBERO downloaded assets into a cache, the configurator detects
`~/.cache/libero/assets`; override it with `--assets` or
`LIBERO_ASSETS_ROOT` when needed.

Validate imports, RAW-VLA computation, and optional headless LIBERO rendering:

```bash
python scripts/setup/check_environment.py rawvla
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python scripts/setup/check_environment.py starvla-libero --render
python scripts/replay/replay_libero_train_rawvla_light_npz.py --help
```

## RoboTwin 2.0

```bash
conda env create -f environment-robotwin2.yml
conda activate robotwin
ROBOTWIN_ASSETS_SOURCE=/path/to/existing/RoboTwin/assets \
  bash scripts/setup/setup_robotwin_assets.sh
# Or omit ROBOTWIN_ASSETS_SOURCE to download and extract the official assets.
ROBOTWIN_PYTHON=python bash third_party/RoboTwin/script/_install.sh
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  python scripts/setup/check_environment.py robotwin --render
python scripts/replay/replay_robotwin2_clean_training.py --help
python scripts/replay/replay_robotwin2_paired_lighting_training.py --help
```

The RoboTwin installer builds CuRobo and applies the SAPIEN/MPLib compatibility
patches required by the pinned RoboTwin checkout. Simulator assets are not
stored in Git and require about 30 GB. The
`scripts/setup/setup_robotwin_assets.sh` script either links an existing
complete asset tree or downloads and extracts the three official archives.
Keep the activated environment's `bin` directory on `PATH`: CuRobo
uses the environment's `ninja` executable when it must rebuild CUDA extensions.

## FastWAM

FastWAM uses a newer PyTorch/CUDA stack and should not share the StarVLA env.

```bash
conda env create -f environment-fastwam.yml
conda activate fastwam
python -m pip install -e ./third_party/FastWAM
python scripts/setup/check_environment.py fastwam
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
python scripts/setup/check_environment.py openvla-oft
```

## Isaac GR00T

The GR00T submodule owns an `uv.lock`, so its locked `uv` environment is the
authority. The top-level Conda file is only a Python/system-tool bootstrap.

```bash
conda env create -f environment-groot.yml
conda activate groot
cd third_party/Isaac-GR00T
uv sync --frozen
uv run python ../../scripts/setup/check_environment.py groot
```

For architecture-specific NVIDIA platforms, follow the deployment instructions
in the pinned GR00T checkout instead of mixing those dependencies into another
RAW-VLA environment.
