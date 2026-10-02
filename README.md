# RAW-VLA

RAW-VLA is a streaming, illumination-adaptive RAW image frontend for
vision-language-action policies. This repository contains the RAW frontend,
StarVLA integration, RAWVLA-Bench lighting transforms, LIBERO and RoboTwin 2.0
evaluation adapters, and the replay code used to produce training trajectories.

## Repository layout

```text
baselines/rawvla/          RAW-VLA model and architecture notes
benchmark/rawvla-bench/   RAW formation, lighting domains, and frozen manifests
starVLA/                   StarVLA integration (submodule)
LIBERO-git/                LIBERO integration (submodule)
third_party/RoboTwin/      RoboTwin 2.0 integration (submodule)
third_party/openvla-oft/   OpenVLA-OFT dependency (submodule)
third_party/FastWAM/       FastWAM integration (submodule)
third_party/Isaac-GR00T/   GR00T dependency (submodule)
scripts/                   training, evaluation, replay, and release checks
replay/                    replay documentation and small manifests
finetune/calibration/      default-ISP calibration utilities
```

## Clone

```bash
git clone --recurse-submodules https://github.com/ShuhongLL/RawVLA.git
cd RawVLA
git submodule sync --recursive
git submodule update --init --recursive
python scripts/check_release.py
```

Several project-maintained submodules are branches of this repository. Their
branch and pinned commit are documented in [SUBMODULES.md](SUBMODULES.md).

## Environments

The full system uses separate environments because StarVLA/LIBERO, RoboTwin,
FastWAM, OpenVLA-OFT, and GR00T have incompatible dependency stacks. Complete
installation and validation commands are in [ENVIRONMENTS.md](ENVIRONMENTS.md).

The two main environments are:

```bash
conda env create -f environment-starvla-libero.yml
conda env create -f environment-robotwin2.yml
```

After installing StarVLA/LIBERO, generate the ignored machine-local LIBERO
configuration with `python scripts/configure_libero.py`.

## Model smoke test

With PyTorch installed:

```bash
python -m baselines.rawvla.smoke_test
```

This validates RAW-VLA tensor contracts, recurrent-state updates, theta
parameterization, and gradients. The current split architecture decodes theta
from `[fused current feature, updated recurrent state]`; the descriptor is not
concatenated into the decoder twice.

## Checkpoints

Download the public StarVLA LIBERO checkpoints and their base models with:

```bash
python download_starvla_libero.py
# Add --all for every listed model family.
python test_starvla_libero.py
```

Set `STARVLA_ROOT=/custom/path/to/starVLA` when the StarVLA checkout is not at
the repository default.

## Training RAW-VLA

The public Qwen3-OFT/LIBERO example is
[`configs/rawvla_qwen3_oft_libero.yaml`](configs/rawvla_qwen3_oft_libero.yaml).
Paths are resolved through `RAWVLA_ROOT`:

```bash
export RAWVLA_ROOT="$PWD"
conda activate starVLA
cd starVLA
accelerate launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ../configs/rawvla_qwen3_oft_libero.yaml
```

The VLA backbone is frozen and RAW-VLA is initialized from scratch. Adjust the
dataset/checkpoint paths or loss weights in the config before a production run.

## Replay data

Replay source code is part of the release. See [replay/README.md](replay/README.md)
for data downloads and smoke commands for:

- LIBERO simulator re-rendering into paired RAW/RGB NPZ trajectories;
- RoboTwin 2.0 clean expert replay;
- RoboTwin 2.0 paired clean/random-light state-copy replay;
- conversion of existing LIBERO RLDS shards into RAWVLA-Light NPZ data.

Generated trajectories, datasets, checkpoints, and videos are intentionally
excluded by `.gitignore`.

## Evaluation

The main public runners derive paths from the checkout and accept overrides via
environment variables:

```bash
PY=python bash benchmark/rawvla-bench/scripts/run_libero_bench.sh

DATA_ROOT=/path/to/data \
CKPT_ROOT=/path/to/checkpoints \
bash scripts/run_robotwin2_starvla_rgb_eval.sh smoke
```

Run scripts with `--help` where supported and start with one task/episode before
launching full benchmark matrices.

## Public-release policy

Do not commit datasets, checkpoints, generated media, result ledgers, cluster
submission credentials, or machine-specific environment activation scripts.
Before publishing, run:

```bash
python scripts/check_release.py --strict
```

The root project is released under the MIT License. Third-party submodules
retain their own licenses.
