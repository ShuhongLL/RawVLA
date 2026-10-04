# RoboTwin 2.0 RAW-VLA training configurations

The main RoboTwin 2.0 recipes are [`pi0.yaml`](pi0.yaml) and
[`pi05.yaml`](pi05.yaml). Both inherit
[`rawvla_robotwin.yaml`](../base/rawvla_robotwin.yaml), which configures the
three-view paired RAW/RGB NPZ loader and the shared RAW-VLA frontend.

Use the published RawVLA-Bench training trajectories from the root README;
simulator replay is not required for training. RAW-VLA training runs in the
StarVLA environment (`environment-libero.yml`, Conda name `starvla-libero`).
The separate `robotwin` environment is for simulator evaluation.

Prepare the frozen π0/π0.5 policy checkpoints and their original normalization
statistics as described in
[`experiments/simulation/robotwin/backbones`](../../experiments/simulation/robotwin/backbones/README.md).
Set `RAWVLA_MODEL_ROOT` to the directory containing `pi0_robotwin`,
`pi0_robotwin_pytorch_45000`, and `pi05_robotwin`. Set
`RAWVLA_TOKENIZER_PATH` to the OpenPI-compatible `paligemma_tokenizer.model`.

From the repository root, run one of the following:

```bash
export RAWVLA_ROOT="$PWD"
export RAWVLA_MODEL_ROOT=/path/to/robotwin2_backbones
export RAWVLA_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model
conda activate starvla-libero
cd starVLA
accelerate launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ../configs/robotwin/pi0.yaml

# Or train π0.5:
accelerate launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ../configs/robotwin/pi05.yaml
```

π0 uses the original 12-arm-component delta-action transform; π0.5 uses
absolute actions. Both use each policy's own downloaded normalization stats.
The recipes initialize the current RAW-VLA frontend from scratch. The
historical paper checkpoints followed multi-stage refinement and predate the
current frontend changes, so these single-run recipes are not bitwise
reproductions of those weights.
