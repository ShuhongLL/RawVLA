# Checkpoint Inventory

This document records the local checkpoint/model paths needed to reproduce the
RAW-VLA benchmark runs on another machine.

Main workspace:

```text
/path/to/RAW-VLA
```

## Quick Map

```text
RAW-VLA/
├── starVLA/playground/Pretrained_models/StarVLA/
│   ├── Qwen*-LIBERO-4in1/
│   ├── WM4A-*-LIBERO-4in1/
│   ├── Qwen3-VL-OFT-RoboTwin2-All/
│   └── Qwen3-VL-OFT-Robotwin2/
├── starvla_robotwin2_checkpoints/
│   ├── Qwen3-VL-OFT-RoboTwin2-All/
│   └── Qwen3-VL-OFT-Robotwin2/
├── fastwam_robotwin2_checkpoint/
│   ├── robotwin_uncond_3cam_384.pt
│   └── robotwin_uncond_3cam_384_dataset_stats.json
└── fastwam_model_cache/
    ├── DiffSynth-Studio/
    └── Wan-AI/
```

## Required Evaluation Checkpoints

The main StarVLA / OpenPI evaluation set used by the current LIBERO and
RoboTwin2 runs is:

```text
StarVLA/Qwen3-VL-OFT-LIBERO-4in1
StarVLA/Qwen3-VL-PI-LIBERO-4in1
StarVLA/WM4A-CosmoPredict-GR00T-LIBERO-4in1
StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1
StarVLA/Qwen3-VL-OFT-RoboTwin2-All
tenstep/pi_model_starvla: pi0_libero_starvla
tenstep/pi_model_starvla: pi05_libero_starvla
```

The StarVLA checkpoints are stored under:

```text
/path/to/RAW-VLA/starVLA/playground/Pretrained_models/StarVLA
```

The converted OpenPI PI0 / PI0.5 LIBERO checkpoints are stored under:

```text
/path/to/RAW-VLA/openpi_converted_protocol
```

The preferred converted OpenPI precision for evaluation is `bfloat16`:

```text
openpi_converted_protocol/pi0_libero_starvla/bfloat16/model.safetensors
openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors
```

## Download Commands

Download the StarVLA evaluation checkpoint repos:

```bash
PRETRAINED=/path/to/RAW-VLA/starVLA/playground/Pretrained_models

huggingface-cli download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 \
  --local-dir "$PRETRAINED/StarVLA/Qwen3-VL-OFT-LIBERO-4in1" \
  --max-workers 4

huggingface-cli download StarVLA/Qwen3-VL-PI-LIBERO-4in1 \
  --local-dir "$PRETRAINED/StarVLA/Qwen3-VL-PI-LIBERO-4in1" \
  --max-workers 4

huggingface-cli download StarVLA/WM4A-CosmoPredict-GR00T-LIBERO-4in1 \
  --local-dir "$PRETRAINED/StarVLA/WM4A-CosmoPredict-GR00T-LIBERO-4in1" \
  --max-workers 4

huggingface-cli download StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1 \
  --local-dir "$PRETRAINED/StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1" \
  --max-workers 4

huggingface-cli download StarVLA/Qwen3-VL-OFT-RoboTwin2-All \
  --local-dir "$PRETRAINED/StarVLA/Qwen3-VL-OFT-RoboTwin2-All" \
  --max-workers 4
```

Download the base VLM and world-model dependencies used by those checkpoints:

```bash
PRETRAINED=/path/to/RAW-VLA/starVLA/playground/Pretrained_models

huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$PRETRAINED/Qwen3-VL-4B-Instruct" \
  --max-workers 4

huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World \
  --local-dir "$PRETRAINED/nvidia/Cosmos-Predict2-2B-Video2World" \
  --max-workers 4

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir "$PRETRAINED/Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
  --max-workers 4
```

Download the converted OpenPI PI0 / PI0.5 LIBERO checkpoints:

```bash
OPENPI_ROOT=/path/to/RAW-VLA/openpi_converted_protocol

huggingface-cli download tenstep/pi_model_starvla \
  --local-dir "$OPENPI_ROOT" \
  --include \
    "pi0_libero_starvla/bfloat16/*" \
    "pi0_libero_starvla/bfloat16/assets/physical-intelligence/libero/*" \
    "pi05_libero_starvla/bfloat16/*" \
    "pi05_libero_starvla/bfloat16/assets/physical-intelligence/libero/*" \
    "paligemma_tokenizer.model" \
  --max-workers 4
```

If the large OpenPI safetensors download is unstable, use the local resumable
helper that was used for the existing copy:

```bash
/path/to/RAW-VLA/download_openpi_libero_starvla_bf16.sh
```

## StarVLA LIBERO Checkpoints

Root:

```text
/path/to/RAW-VLA/starVLA/playground/Pretrained_models/StarVLA
```

Relevant LIBERO finetuned checkpoints:

```text
Qwen2.5-VL-OFT-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
Qwen2.5-VL-FAST-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt
WM4A-CosmoPredict-GR00T-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
WM4A-Wan2d2-OFT-LIBERO-4in1/checkpoints/steps_60000_pytorch_model.pt
```

Each model directory also has a `dataset_statistics.json` file used for action
normalization/unnormalization.

Base VLM / world-model dependencies are also under:

```text
/path/to/RAW-VLA/starVLA/playground/Pretrained_models
```

Examples:

```text
Qwen2.5-VL-3B-Instruct/
Qwen3-VL-4B-Instruct/
StarVLA/Qwen2.5-VL-3B-Instruct-Action/
nvidia/Cosmos-Predict2-2B-Video2World/
Wan-AI/Wan2.2-TI2V-5B-Diffusers/
```

## StarVLA RoboTwin2 Checkpoints

Primary local copy:

```text
/path/to/RAW-VLA/starvla_robotwin2_checkpoints
```

Files:

```text
Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt
Qwen3-VL-OFT-RoboTwin2-All/dataset_statistics.json

Qwen3-VL-OFT-Robotwin2/checkpoints/steps_40000_pytorch_model.pt
Qwen3-VL-OFT-Robotwin2/dataset_statistics.json
```

Sizes observed on 2026-08-17:

```text
Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt  9.2G
Qwen3-VL-OFT-Robotwin2/checkpoints/steps_40000_pytorch_model.pt       9.2G
```

The same RoboTwin2 checkpoints are also present under the StarVLA standard
pretrained-model root:

```text
/path/to/RAW-VLA/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt
/path/to/RAW-VLA/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-Robotwin2/checkpoints/steps_40000_pytorch_model.pt
```

For RoboTwin2 RGB baseline/eval reproduction, the preferred model is usually:

```text
Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt
```

## FastWAM RoboTwin2 Checkpoint

Root:

```text
/path/to/RAW-VLA/fastwam_robotwin2_checkpoint
```

Files:

```text
robotwin_uncond_3cam_384.pt
robotwin_uncond_3cam_384_dataset_stats.json
```

Sizes observed on 2026-08-17:

```text
robotwin_uncond_3cam_384.pt  12G
```

This is the main FastWAM checkpoint used by the RoboTwin2 FastWAM runs.

## FastWAM Model Cache

Root:

```text
/path/to/RAW-VLA/fastwam_model_cache
```

Important cached dependencies include:

```text
DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors
DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
Wan-AI/Wan2.1-T2V-1.3B/
```

These files are model dependencies/cache for FastWAM. The actual RoboTwin2
FastWAM policy checkpoint is in `fastwam_robotwin2_checkpoint/`.

## Existing Scattered Docs

Some checkpoint information also appears in the repository docs:

```text
/path/to/RAW-VLA/README.md
/path/to/RAW-VLA/ROBOTWIN2_SETUP.md
/path/to/RAW-VLA/docs/robotwin2_rgb_baseline_results.md
/path/to/RAW-VLA/third_party/FastWAM/README.md
```

This file is the centralized checkpoint/path inventory for data and environment
transfer.
