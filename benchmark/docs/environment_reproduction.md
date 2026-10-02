# Environment Reproduction Notes

This note records the current status of the StarVLA LIBERO and RoboTwin2 test
environments.

## Short Answer

The top-level files below are useful starting points, but they are not complete
authoritative reproductions of the environments that have actually been used on
this machine:

```text
/path/to/RAW-VLA/environment-starvla-libero.yml
/path/to/RAW-VLA/environment-robotwin2.yml
```

They are hand-maintained environment specs and currently differ from the
working environments observed on this host. Do not rely on them alone when
building another machine.

Actual environment snapshots have been exported here:

```text
/path/to/RAW-VLA/benchmark/docs/env_repro/
├── starvla_root_env_export.yml
├── starvla_root_conda_explicit.txt
├── starvla_root_pip_freeze.txt
├── starvla_user_pip_freeze.txt
├── robotwin_root_env_export.yml
├── robotwin_root_conda_explicit.txt
└── robotwin_root_pip_freeze.txt
```

## Observed Working Environments

### StarVLA / LIBERO

Working Python checked:

```text
/path/to/conda_envs/starVLA/bin/python
```

Important observed versions:

```text
python:       3.10.20
torch:        2.6.0+cu124
torchvision:  0.21.0+cu124
transformers: 4.57.0
accelerate:   1.5.2
numpy:        1.26.4
mujoco:       3.2.3
robosuite:    1.4.0
bddl:         1.0.1
gym:          0.25.2
opencv:       4.11.0
qwen-vl-utils: installed
```

The user EFS StarVLA env also works on this host:

```text
/path/to/conda_envs/starVLA/bin/python
```

It has the same key StarVLA/LIBERO package versions and includes `h5py==3.16.0`.
The helper activation script is:

```text
/path/to/conda_envs/activate_starVLA.sh
```

That script falls back to image Python plus extra site-packages if the packed
EFS conda Python cannot run on the container's glibc. This fallback behavior is
not represented in a conda YAML file.

### RoboTwin2

Working Python checked:

```text
/path/to/conda_envs/robotwin/bin/python
```

Important observed versions:

```text
python:       3.10.20
torch:        2.6.0+cu124
torchvision:  0.21.0+cu124
sapien:       3.0.0b1
mplib:        0.2.1
transforms3d: 0.4.2
scipy:        1.10.1
scikit-image: 0.21.0
gymnasium:    0.29.1
trimesh:      4.4.3
open3d:       0.18.0
imageio:      2.34.2
toppra:       0.6.3
opencv:       4.11.0
curobo:       editable nvidia-curobo from NVlabs/curobo
warp-lang:    1.12.0
pydantic:     2.13.4
```

The user EFS path below is present but was not a standalone working Python in
this container:

```text
/path/to/conda_envs/robotwin
```

Directly running that Python raised `ModuleNotFoundError: No module named
site`, so it should not be copied as the sole reproducibility source.

## Differences From Top-Level YAMLs

Current top-level YAMLs mention CUDA 13 / `torch==2.13.0+cu130`, but the
working root environments observed here use:

```text
torch==2.6.0+cu124
torchvision==0.21.0+cu124
```

RoboTwin2 also depends on editable/source pieces and runtime patches that are
not fully captured by plain YAML:

```text
nvidia-curobo editable install from NVlabs/curobo
RoboTwin benchmark checkout and local StarVLA interface patches
RoboTwin assets under the benchmark checkout
Vulkan/NVIDIA driver availability for SAPIEN
```

## Recommended Reproduction Workflow

For another similar Linux x86_64 CUDA machine, start from the exported files:

```bash
conda env create -f /path/to/RAW-VLA/benchmark/docs/env_repro/starvla_root_env_export.yml
conda env create -f /path/to/RAW-VLA/benchmark/docs/env_repro/robotwin_root_env_export.yml
```

If conda resolution drifts, use the explicit specs as a stronger starting point
for conda packages:

```bash
conda create -n starVLA --file /path/to/RAW-VLA/benchmark/docs/env_repro/starvla_root_conda_explicit.txt
conda create -n robotwin --file /path/to/RAW-VLA/benchmark/docs/env_repro/robotwin_root_conda_explicit.txt
```

Then use the corresponding pip freeze files to check package differences:

```text
starvla_root_pip_freeze.txt
robotwin_root_pip_freeze.txt
```

Important: after creating the env, also install or expose the source trees used
by this workspace:

```text
/path/to/RAW-VLA/starVLA
/path/to/RAW-VLA/third_party/RoboTwin
/path/to/RAW-VLA/RoboTwin or /path/to/RAW-VLA/RoboTwin
```

For RoboTwin2, make sure the benchmark assets and data paths are also present.
See:

```text
/path/to/RAW-VLA/benchmark/docs/benchmark_data_inventory.md
```

## Practical Guidance

Use the top-level `environment-*.yml` files as readable notes only. For a real
machine migration, use `benchmark/docs/env_repro/*` plus this document, then run
smoke tests for:

```text
StarVLA LIBERO import/model-load path
LIBERO simulator import/render path
RoboTwin2 SAPIEN/MPLib/CuRobo import path
RoboTwin2 StarVLA evaluation wrapper
```
