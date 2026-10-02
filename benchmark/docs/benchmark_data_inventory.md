# Benchmark Data Inventory

This document records where the downloaded benchmark data lives and what each
directory contains.

## Root

Data root:

```text
/path/to/RAW-VLA/benchmark_data
```

Current top-level layout:

```text
benchmark_data/
├── libero/               # Main LIBERO data, about 278G
├── robotwin2/            # RoboTwin2 aloha-agilex clean_50 data, about 69G
├── libero_plus_assets/   # LIBERO-plus assets archive, about 6.0G
└── libero_plus_light/    # Small LIBERO-plus light task range metadata
```

Total size is about 352G.

## Download Commands

The helper script below downloads the LIBERO HDF5 evaluation data and the
RoboTwin2 `aloha-agilex_clean_50` evaluation split to the paths documented in
this file:

```bash
/path/to/RAW-VLA/scripts/download_replay_datasets.sh
```

Equivalent manual Hugging Face downloads:

```bash
hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir /path/to/RAW-VLA/benchmark_data/libero/LIBERO-datasets \
  --include "libero_10/*.hdf5" "libero_goal/*.hdf5" "libero_object/*.hdf5" "libero_spatial/*.hdf5" "README.md" ".gitattributes"

hf download TianxingChen/RoboTwin2.0 \
  --repo-type dataset \
  --local-dir /path/to/RAW-VLA/benchmark_data/robotwin2 \
  --include "dataset/*/aloha-agilex_clean_50.zip"
```

## LIBERO

Root:

```text
/path/to/RAW-VLA/benchmark_data/libero
```

Main subdirectories:

```text
libero/
├── LIBERO-datasets/                    # Official/raw HDF5 demonstrations
├── modified_libero_rlds/               # OpenVLA/RLDS TFRecord train data
├── rawvla_light_train_raw_npz_action_xmlfix_20260816T0839Z/
│                                          # Active RAW training NPZ data
└── rawvla_light_train_rgb_replay_npz_action_xmlfix_20260816T0839Z/
                                           # Active RGB replay training NPZ data
```

### Official HDF5 Data

Path:

```text
/path/to/RAW-VLA/benchmark_data/libero/LIBERO-datasets
```

This is the official/raw LIBERO demonstration data in `.hdf5` format.

Counts observed:

```text
libero_10:      10 hdf5 files
libero_90:      48 hdf5 files
libero_goal:    10 hdf5 files
libero_object:  10 hdf5 files
libero_spatial: 10 hdf5 files
```

Example:

```text
LIBERO-datasets/libero_spatial/
└── pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5
```

### Modified RLDS Data

Path:

```text
/path/to/RAW-VLA/benchmark_data/libero/modified_libero_rlds
```

This is OpenVLA/RLDS-style training data stored as TFRecords.

Layout:

```text
modified_libero_rlds/
├── libero_10_no_noops/
├── libero_goal_no_noops/
├── libero_object_no_noops/
└── libero_spatial_no_noops/
```

Example:

```text
modified_libero_rlds/libero_spatial_no_noops/1.0.0/
├── dataset_info.json
├── features.json
└── libero_spatial-train.tfrecord-00000-of-00016
```

### RAW-VLA / StarVLA Training Data

Paths:

```text
/path/to/RAW-VLA/benchmark_data/libero/rawvla_light_train_raw_npz_action_xmlfix_20260816T0839Z
/path/to/RAW-VLA/benchmark_data/libero/rawvla_light_train_rgb_replay_npz_action_xmlfix_20260816T0839Z
```

These are the RAW-VLA/StarVLA training datasets derived from LIBERO. They are
organized in LeRobot-like subset directories and store episode variants as
`.npz` files.

Shared subset layout:

```text
rawvla_light_train_*_npz_action_xmlfix_20260816T0839Z/
├── _manifests/
├── libero_10_no_noops_1.0.0_lerobot/
├── libero_goal_no_noops_1.0.0_lerobot/
├── libero_object_no_noops_1.0.0_lerobot/
└── libero_spatial_no_noops_1.0.0_lerobot/
```

Example:

```text
rawvla_light_train_raw_npz_action_xmlfix_20260816T0839Z/libero_spatial_no_noops_1.0.0_lerobot/
├── episode_000000/variant_00.npz
├── episode_000001/variant_00.npz
└── ...
```

Observed file counts:

```text
rawvla_light_train_raw_npz_action_xmlfix_20260816T0839Z:
  _manifests: 16 files
  libero_10_no_noops_1.0.0_lerobot: 399 NPZ files
  libero_goal_no_noops_1.0.0_lerobot: 456 NPZ files
  libero_object_no_noops_1.0.0_lerobot: 462 NPZ files
  libero_spatial_no_noops_1.0.0_lerobot: 454 NPZ files
  total: 1771 NPZ files

rawvla_light_train_rgb_replay_npz_action_xmlfix_20260816T0839Z:
  _manifests: 16 files
  libero_10_no_noops_1.0.0_lerobot: 399 NPZ files
  libero_goal_no_noops_1.0.0_lerobot: 456 NPZ files
  libero_object_no_noops_1.0.0_lerobot: 462 NPZ files
  libero_spatial_no_noops_1.0.0_lerobot: 454 NPZ files
  total: 1771 NPZ files
```

This action-XML-fixed pair is the only active RAW-VLA/StarVLA train replay.
Training configurations must pin
`dataset_version: action_xmlfix_20260816T0839Z`. The old unversioned roots and
quarantine directories were deleted on 2026-08-22 because their replay omitted
some per-demo fixtures/assets.

### Quarantine Directories

The following failed/quarantined replay directories were debugging artifacts,
not training datasets, and were deleted on 2026-08-22:

```text
libero/failed_replay_51_quarantine_20260814T1155Z/
libero/failed_starvla_protocol_42_quarantine_20260814T1350Z/
```

## RoboTwin2

Root:

```text
/path/to/RAW-VLA/benchmark_data/robotwin2
```

Main layout:

```text
robotwin2/
├── dataset/      # Main downloaded and extracted RoboTwin2 data
└── .cache/       # Hugging Face download cache and old incomplete files
```

The `.cache` directory is not required for normal dataset loading.

### Downloaded Split

The currently downloaded split is the official RoboTwin2 `aloha-agilex_clean_50`
data for 50 tasks:

```text
dataset/*/aloha-agilex_clean_50.zip
```

Each zip has been extracted in place.

Observed status:

```text
aloha-agilex_clean_50.zip files: 50
extracted aloha-agilex_clean_50 directories: 50
trajectory pkl files: 2500
episode hdf5 files: 2500
```

Verification on 2026-08-17:

```text
Hugging Face dataset task directories: 50
remote aloha-agilex_clean_50.zip entries: 50
local aloha-agilex_clean_50.zip files: 50
remote/local zip byte sum: 23780715316
bad zip central directories: 0
extracted tasks with 50 hdf5 + 50 pkl files: 50
```

Source checked:

```text
https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset
```

Task-level layout:

```text
robotwin2/dataset/<task>/
├── aloha-agilex_clean_50.zip
└── aloha-agilex_clean_50/
    ├── _traj_data/
    │   ├── episode0.pkl
    │   ├── episode1.pkl
    │   └── ...
    └── data/
        ├── episode0.hdf5
        ├── episode1.hdf5
        └── ...
```

Example:

```text
robotwin2/dataset/adjust_bottle/aloha-agilex_clean_50/
├── _traj_data/episode0.pkl
└── data/episode0.hdf5
```

Note: this is not the full RoboTwin2 dataset across all embodiments and
randomized settings. It is the 50-task `aloha-agilex_clean_50` subset requested
for local use.

## LIBERO-plus Assets

Path:

```text
/path/to/RAW-VLA/benchmark_data/libero_plus_assets
```

Contents:

```text
libero_plus_assets/
└── assets.zip
```

The archive is about 6.4GB as a file, and the directory reports about 6.0G via
`du -h`.

## LIBERO-plus Light Metadata

Path:

```text
/path/to/RAW-VLA/benchmark_data/libero_plus_light
```

Contents observed:

```text
libero_plus_light/
└── light_task_ranges.tsv
```

This is small metadata/configuration, not a large dataset.

## Quick Commands

Check top-level sizes:

```bash
du -h --max-depth=2 /path/to/RAW-VLA/benchmark_data | sort -h
```

Count RoboTwin2 clean_50 zips and extracted directories:

```bash
BASE=/path/to/RAW-VLA/benchmark_data/robotwin2/dataset
find "$BASE" -type f -name 'aloha-agilex_clean_50.zip' | wc -l
find "$BASE" -mindepth 2 -maxdepth 2 -type d -name 'aloha-agilex_clean_50' | wc -l
```

Count LIBERO official HDF5 files:

```bash
BASE=/path/to/RAW-VLA/benchmark_data/libero/LIBERO-datasets
for d in "$BASE"/libero_*; do
  printf '%s ' "$(basename "$d")"
  find "$d" -type f -name '*.hdf5' | wc -l
done
```
