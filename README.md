<div align="center">

<h1>📷 RawVLA: Embodied Neural Image Signal Processor for Robotic Manipulation</h1>

<img src="assets/teaser.png" alt="Figure 1 from the RawVLA paper: method, benchmark, and real-world results" width="100%">

<p>
  <a href="https://shuhongll.github.io/">Shuhong Liu</a><sup>1,2</sup>,
  Heng Zhou<sup>2</sup>, Lingfeng Qian<sup>2</sup>, Yuhao Fang<sup>2</sup>,
  Xianbao Hou<sup>2</sup>, Qianyu Zhou<sup>1</sup>,
  <a href="https://sites.google.com/view/linguedu/home">Lin Gu</a><sup>4</sup>,
  Wei Sui<sup>2</sup>, <a href="https://marsyang.site/">Jianfei Yang</a><sup>3</sup>,
  <a href="https://cuiziteng.github.io/">Ziteng Cui</a><sup>1,5</sup>
</p>

<p>
  <sup>1</sup>The University of Tokyo &nbsp;
  <sup>2</sup>D-Robotics &nbsp;
  <sup>3</sup>NTU &nbsp;
  <sup>4</sup>Tohoku University &nbsp;
  <sup>5</sup>HKUST(GZ)
</p>

<p>
  <a href="https://shuhongll.github.io/rawvla/">Project Page</a> ·
  <a href="https://arxiv.org/abs/2609.37530">Paper</a> ·
  <a href="https://github.com/ShuhongLL/RawVLA">Code</a> ·
  <a href="https://huggingface.co/datasets/ToferFish/RawVLA-Bench">RawVLA-Bench</a>
</p>

</div>

RAW-VLA is a streaming, illumination-adaptive RAW image frontend for
vision-language-action policies. This repository contains the RAW frontend,
StarVLA integration, RAWVLA-Bench lighting transforms, LIBERO and RoboTwin 2.0
evaluation adapters, and the replay code used to produce training trajectories.

## 📦 Installation

Clone the repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/ShuhongLL/RawVLA.git
cd RawVLA
git submodule sync --recursive
git submodule update --init --recursive
```

Several project-maintained submodules are branches of this repository. Their
branch and pinned commit are documented in [SUBMODULES.md](SUBMODULES.md).

### Conda environments

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

## 🧠 VLA Checkpoints

Download the public StarVLA LIBERO checkpoints and their base models with:

```bash
python tools/checkpoints/download_starvla_libero.py
# Add --all for every listed model family.
python tools/checkpoints/validate_starvla_libero.py
```

Set `STARVLA_ROOT=/custom/path/to/starVLA` when the StarVLA checkout is not at
the repository default.

## 🗂️ RawVLA-Bench

### Training Data

The paired RAW/RGB training replay data is available directly from
[RawVLA-Bench on Hugging Face](https://huggingface.co/datasets/ToferFish/RawVLA-Bench).
It contains 2,403 successful trajectories: 1,771 from LIBERO and 632 from
RoboTwin 2.0. No simulator replay is needed to use the published data.

```bash
hf download ToferFish/RawVLA-Bench \
  --repo-type dataset \
  --local-dir benchmark_data/RawVLA-Bench
```

The LIBERO RAW files then live under
`benchmark_data/RawVLA-Bench/data/libero/raw`; the paired RGB files and
RoboTwin data are in the same download. To regenerate the trajectories from
source demonstrations, see [replay/README.md](replay/README.md).

### Evaluation Manifests

The evaluation protocol is recorded in frozen simulator rollout manifests,
not pre-rendered test trajectories. These JSON files are included in this
GitHub repository, not in the Hugging Face training-data release:

- [LIBERO full evaluation](experiments/simulation/libero/manifests/libero_manifest_50_init_states_10000_rollouts.json):
  40 tasks × 50 initial states × 5 lighting domains = 10,000 rollouts.
- [LIBERO smaller evaluation](experiments/simulation/libero/manifests/libero_manifest_10_init_states_2000_rollouts.json):
  40 tasks × 10 initial states × 5 lighting domains = 2,000 rollouts.
- [RoboTwin 2.0 evaluation](experiments/simulation/robotwin/manifests/robotwin2_test_manifest_50_seeds_3250_rollouts.json):
  13 tasks × 50 seeds × 5 lighting domains = 3,250 rollouts.

The manifests fix episode seeds and lighting conditions. The LIBERO runner
reads its manifest directly; the current RoboTwin runner does not load its
JSON manifest automatically. During evaluation, the simulator renders
observations and the policy produces actions.

## 🏋️ Training

The public LIBERO configurations share
[`configs/base/rawvla_libero.yaml`](configs/base/rawvla_libero.yaml) and contain
complete overrides for Qwen3-OFT, Qwen3-PI, WM4A-Cosmos, WM4A-Wan, PI0, and
PI0.5 under [`configs/libero`](configs/libero). Paths are resolved through
`RAWVLA_ROOT`. This example uses the published LIBERO replay data described
above:

```bash
export RAWVLA_ROOT="$PWD"
export LIBERO_CONFIG_PATH="$RAWVLA_ROOT/.local/libero"
conda activate starvla-libero
cd starVLA
accelerate launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ../configs/libero/qwen3_oft.yaml \
  --datasets.vla_data.cache_root "$RAWVLA_ROOT/benchmark_data/RawVLA-Bench/data/libero/raw"
```

Each child config uses a relative `extends` entry. The training loader merges
the base first and the selected backbone config second; command-line dotlist
overrides remain highest priority. The VLA backbone is frozen and RAW-VLA is
initialized from scratch.

## 📊 Evaluation

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

## 📚 Citation

If you use RawVLA or RawVLA-Bench, please cite the paper:

```bibtex
@article{liu2026rawvla,
  title={RawVLA: Embodied Neural Image Signal Processor for Robotic Manipulation},
  author={Liu, Shuhong and Zhou, Heng and Qian, Lingfeng and Fang, Yuhao and Hou, Xianbao and Zhou, Qianyu and Gu, Lin and Sui, Wei and Yang, Jianfei and Cui, Ziteng},
  journal={arXiv preprint arXiv:2609.37530},
  year={2026},
  url={https://arxiv.org/abs/2609.37530}
}
```
