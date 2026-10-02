# RoboTwin 2.0 π0 and π0.5 backbones

This directory contains the reproducible download and conversion entrypoints for
the two frozen policy backbones used to train RAWVLA on RoboTwin 2.0. Model
weights are intentionally stored outside this Git repository.

## Provenance

| Backbone | Source | Pinned revision | Downloaded format | Extra step for RAWVLA |
|---|---|---|---|---|
| π0 | `Heisen0928/pi0_robotwin`, checkpoint `45000` | `f163ea4a7d4b4806a811644117e2e4e6ab29e1a9` | JAX/Orbax | Convert weights to PyTorch |
| π0.5 | `SidneyXie/pi05_robotwin` | `e49e2ab6c11f07511573b67261bd129e88d0a416` | PyTorch/LeRobot | Export policy normalization to JSON |

The π0 checkpoint is a full finetune, not a LoRA checkpoint. The official
OpenPI JAX-to-PyTorch converter is not LoRA-aware and must not be used for LoRA
checkpoints.

## Prerequisites

Install the Hugging Face CLI in the download environment:

```bash
python -m pip install 'huggingface_hub[cli]'
```

Define an asset root outside the source tree:

```bash
export MODEL_ROOT=/path/to/robotwin2_backbones
mkdir -p "$MODEL_ROOT"
```

The scripts accept an explicit destination, so they do not depend on a
particular cluster filesystem layout.

## π0: download and convert JAX/Orbax weights

Download the inference/conversion subset of the pinned checkpoint:

```bash
bash backbone/robotwin/download_pi0.sh "$MODEL_ROOT/pi0_robotwin"
```

Expected source layout:

```text
pi0_robotwin/
└── 45000/
    ├── _CHECKPOINT_METADATA
    ├── params/
    └── assets/
        └── robotwin_50_clean/
            └── norm_stats.json
```

RAWVLA training uses a PyTorch implementation of the frozen π0 backbone, so
the Orbax weights must be converted:

```bash
export OPENPI_ROOT=/path/to/a/compatible/openpi
export OPENPI_PYTHON=/path/to/openpi-python

bash backbone/robotwin/convert_pi0_jax_to_pytorch.sh \
  "$MODEL_ROOT/pi0_robotwin/45000" \
  "$MODEL_ROOT/pi0_robotwin_pytorch_45000"
```

`OPENPI_ROOT` must contain both:

```text
examples/convert_jax_model_to_pytorch.py
src/openpi/training/config.py
```

and its config registry must contain `pi0_base_aloha_robotwin_full`. Use the
official OpenPI converter from the same compatible OpenPI checkout; do not copy
a converter from an unrelated revision into the runtime environment. The
converter invocation follows the official OpenPI interface:

```text
--checkpoint_dir <orbax-checkpoint>
--config_name pi0_base_aloha_robotwin_full
--output_path <pytorch-checkpoint>
--precision bfloat16
```

Expected converted layout:

```text
pi0_robotwin_pytorch_45000/
├── config.json
└── model.safetensors
```

Do not recompute π0 normalization statistics. Keep using the original policy
asset:

```text
pi0_robotwin/45000/assets/robotwin_50_clean/norm_stats.json
```

The RAWVLA training configuration therefore uses separate paths for the
converted model and the original policy statistics.

## π0.5: download PyTorch weights and export policy statistics

Download the complete pinned LeRobot policy:

```bash
bash backbone/robotwin/download_pi05.sh "$MODEL_ROOT/pi05_robotwin"
```

The downloaded checkpoint already contains PyTorch weights. Do **not** convert
`model.safetensors`. The original files are:

```text
pi05_robotwin/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── train_config.json
```

LeRobot inference consumes the processor JSON and safetensors files directly.
The RAWVLA NPZ loader instead consumes an equivalent JSON representation. Export
it from the checkpoint's preprocessor normalizer:

```bash
python backbone/robotwin/export_pi05_policy_norm_stats.py \
  --checkpoint-dir "$MODEL_ROOT/pi05_robotwin"
```

This adds:

```text
pi05_robotwin/policy_norm_stats.json
```

The exporter copies `action` and `observation.state` statistics; it does not
recompute them from the five-light RAWVLA dataset. The JSON belongs to the
π0.5 policy preprocessing contract, although the RAWVLA loader consumes it.

## Paths used by RAWVLA training

π0:

```yaml
trainer:
  pretrained_checkpoint: /path/to/robotwin2_backbones/pi0_robotwin_pytorch_45000/model.safetensors
datasets:
  vla_data:
    stats_path: /path/to/robotwin2_backbones/pi0_robotwin/45000/assets/robotwin_50_clean/norm_stats.json
```

π0.5:

```yaml
trainer:
  pretrained_checkpoint: /path/to/robotwin2_backbones/pi05_robotwin/model.safetensors
datasets:
  vla_data:
    stats_path: /path/to/robotwin2_backbones/pi05_robotwin/policy_norm_stats.json
```

These policy assets are separate from the trained RAWVLA frontend checkpoint,
for example `steps_100_raw_frontend_pytorch_model.pt`.

## Integrity checks

The scripts validate the required files after each operation. For the verified
local assets used in this project, the normalization files have these SHA-256
values:

```text
7bd3ec58acee3fe73bc6423ab12b904821b05e98d69a68c8a89cc1916208f64b  pi0 norm_stats.json
229a4a2d6cfd1ba4eb2bc0b3aa57f049094032939130e4901485e9ea9910f5f7  pi05 policy_norm_stats.json
```

Different source revisions must not be assumed equivalent merely because their
filenames match.
