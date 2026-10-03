# 📷 RawVLA: Embodied Neural Image Signal Processor for Robotic Manipulation

[Shuhong Liu](https://shuhongll.github.io/), Heng Zhou, Lingfeng Qian,
Yuhao Fang, Xianbao Hou, Qianyu Zhou,
[Lin Gu](https://sites.google.com/view/linguedu/home), Wei Sui,
[Jianfei Yang](https://marsyang.site/), [Ziteng Cui](https://cuiziteng.github.io/)

[Project Page](https://shuhongll.github.io/rawvla/) ·
[Paper](https://arxiv.org/abs/2609.37530) ·
[Code](https://github.com/ShuhongLL/RawVLA) ·
[RawVLA-Bench](https://huggingface.co/datasets/ToferFish/RawVLA-Bench)

RAW-VLA is a streaming, illumination-adaptive RAW image frontend for
vision-language-action policies. This repository contains the RAW frontend,
StarVLA integration, RAWVLA-Bench lighting transforms, LIBERO and RoboTwin 2.0
evaluation adapters, and the replay code used to produce training trajectories.

## Clone

```bash
git clone --recurse-submodules https://github.com/ShuhongLL/RawVLA.git
cd RawVLA
git submodule sync --recursive
git submodule update --init --recursive
```

Several project-maintained submodules are branches of this repository. Their
branch and pinned commit are documented in [SUBMODULES.md](SUBMODULES.md).

## Environments

The full system uses separate environments because StarVLA/LIBERO, RoboTwin,
FastWAM, OpenVLA-OFT, and GR00T have incompatible dependency stacks. Complete
installation and validation commands are in [ENVIRONMENTS.md](ENVIRONMENTS.md).

The two main environments are:

```bash
conda env create -f environment-libero.yml
conda env create -f environment-robotwin2.yml
```

After installing StarVLA/LIBERO, generate the ignored machine-local LIBERO
simulator paths in `.local/libero/config.yaml` with
`python scripts/setup/configure_libero.py`. The versioned RAW-VLA training recipes
live separately in `configs/libero/`.

## Checkpoints

Download the public StarVLA LIBERO checkpoints and their base models with:

```bash
python tools/checkpoints/download_starvla_libero.py
# Add --all for every listed model family.
python tools/checkpoints/validate_starvla_libero.py
```

Set `STARVLA_ROOT=/custom/path/to/starVLA` when the StarVLA checkout is not at
the repository default.

## Training RAW-VLA

The public LIBERO configurations share
[`configs/base/rawvla_libero.yaml`](configs/base/rawvla_libero.yaml) and contain
complete overrides for Qwen3-OFT, Qwen3-PI, WM4A-Cosmos, WM4A-Wan, PI0, and
PI0.5 under [`configs/libero`](configs/libero). Paths are resolved through
`RAWVLA_ROOT`:

```bash
export RAWVLA_ROOT="$PWD"
export LIBERO_CONFIG_PATH="$RAWVLA_ROOT/.local/libero"
conda activate starvla-libero
cd starVLA
accelerate launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ../configs/libero/qwen3_oft.yaml
```

Each child config uses a relative `extends` entry. The training loader merges
the base first and the selected backbone config second; command-line dotlist
overrides remain highest priority. The VLA backbone is frozen and RAW-VLA is
initialized from scratch.

## Replay data

Replay source code is part of the release. See [replay/README.md](replay/README.md)
for data downloads and replay commands for:

- LIBERO simulator re-rendering into paired RAW/RGB NPZ trajectories;
- RoboTwin 2.0 clean expert replay;
- RoboTwin 2.0 paired clean/random-light state-copy replay.

Generated trajectories, datasets, checkpoints, and videos are intentionally
excluded by `.gitignore`.

## Evaluation

The main public runners derive paths from the checkout and accept overrides via
environment variables:

```bash
PY=python bash experiments/simulation/libero/scripts/run_libero_bench.sh

DATA_ROOT=/path/to/data \
CKPT_ROOT=/path/to/checkpoints \
bash scripts/evaluation/run_robotwin2_starvla_rgb_eval.sh one adjust_bottle
```

Run scripts with `--help` where supported and start with one task/episode before
launching full benchmark matrices.

Synthetic EV, bit-depth, chromatic, and tonal studies are intentionally kept
separate from RAWVLA-Bench under
[`experiments/isp_perturbation`](experiments/isp_perturbation/README.md).
Physical-robot entry points are documented under
[`experiments/real_robot`](experiments/real_robot/README.md); RoboTwin and
LIBERO remain simulator integrations.

The root project is released under the MIT License. Third-party submodules
retain their own licenses.
