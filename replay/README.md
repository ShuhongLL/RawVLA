# Replay pipelines

The published paired training trajectories can be downloaded directly from
[RawVLA-Bench on Hugging Face](https://huggingface.co/datasets/ToferFish/RawVLA-Bench).
The workflows below are optional steps for regenerating that data from source
demonstrations.

Replay outputs are intentionally not stored in Git. By default they live under
`benchmark_data/`, which is ignored. Replay commands support explicit
input/output paths; begin with one episode or one worker before scaling up.

## Replay environments (optional)

Use separate simulator environments; neither is needed when using the
published paired trajectories directly. Run the following from the repository
root after initializing its submodules. System GPU, EGL/Vulkan, and simulator
asset requirements are described in [`ENVIRONMENTS.md`](../ENVIRONMENTS.md).

For LIBERO HDF5 replay, install the StarVLA/LIBERO environment. It includes
`h5py` and `bddl`; the configurator creates the ignored machine-local
`.local/libero/config.yaml` used by the simulator:

```bash
conda env create -f environment-libero.yml
conda activate starvla-libero
python -m pip install --no-deps --no-build-isolation -e ./third_party/LIBERO
python -m pip install --no-deps -e ./third_party/openvla-oft
python -m pip install --no-deps -e ./starVLA
export LIBERO_CONFIG_PATH="$PWD/.local/libero"
python scripts/setup/configure_libero.py
python scripts/setup/check_environment.py starvla-libero
python scripts/replay/replay_libero_train_rawvla_light_npz.py --help
```

For RoboTwin clean and paired-lighting replay, use the RoboTwin environment.
The asset script downloads the official simulator assets unless
`ROBOTWIN_ASSETS_SOURCE` points to an existing asset tree:

```bash
conda env create -f environment-robotwin2.yml
conda activate robotwin
bash scripts/setup/setup_robotwin_assets.sh
ROBOTWIN_PYTHON=python bash third_party/RoboTwin/script/_install.sh
python scripts/setup/check_environment.py robotwin
python scripts/replay/replay_robotwin2_clean_training.py --help
python scripts/replay/replay_robotwin2_paired_lighting_training.py --help
```

These two workflows do not require the FastWAM environment. Activate the
matching environment before running each replay command below.

## Download source demonstrations

Install the Hugging Face CLI and run:

```bash
bash scripts/replay/download_replay_datasets.sh
```

Use `DATA_ROOT=/path/to/benchmark_data` to place data outside the checkout.
`PROXY_HTTP` and `PROXY_ALL` are optional; no proxy is enabled by default.

## LIBERO HDF5 simulator replay

This is the canonical paired replay: it restores each demonstration's simulator
state/XML, applies RAWVLA-Bench lighting, and writes target-light RAW plus paired
default-light RGB NPZ files.

```bash
python scripts/replay/replay_libero_train_rawvla_light_npz.py \
  --hdf5-root benchmark_data/libero/LIBERO-datasets \
  --out-root benchmark_data/libero/rawvla_light_train_raw_npz \
  --rgb-out-root benchmark_data/libero/rawvla_light_train_rgb_replay_npz \
  --max-episodes 1 --workers 1
```

Remove `--max-episodes 1` only after inspecting the generated manifest and NPZ
schema. `--replay-mode states` is the reproducible default.

## RoboTwin 2.0 clean expert replay

```bash
python scripts/replay/replay_robotwin2_clean_training.py \
  --robotwin-root third_party/RoboTwin \
  --dataset-root benchmark_data/robotwin2/dataset \
  --output-root benchmark_data/robotwin2/clean_replay \
  --task adjust_bottle --episodes 50 --episode-indices 0 \
  --max-attempts 1
```

The script replays saved joint trajectories through RoboTwin physics and checks
render fidelity against the original training frames. Inputs are explicit CLI
arguments; start with one selected episode. The dataset downloader extracts the
downloaded task archives into the layout expected by this command.

## RoboTwin 2.0 paired-lighting replay

After successful clean replays, freeze their lighting assignments in a manifest,
then run the paired replay against it:

```bash
python scripts/replay/build_robotwin2_lighting_train_manifest.py \
  --clean-replay-root benchmark_data/robotwin2/clean_replay \
  --dataset-root benchmark_data/robotwin2/dataset \
  --output benchmark_data/robotwin2/lighting_train_manifest.json
python scripts/replay/replay_robotwin2_paired_lighting_training.py \
  --robotwin-root third_party/RoboTwin \
  --manifest benchmark_data/robotwin2/lighting_train_manifest.json \
  --output-root benchmark_data/robotwin2/paired_lighting_replay \
  --num-workers 1 --worker-index 0
python scripts/replay/replay_robotwin2_paired_lighting_training.py \
  --robotwin-root third_party/RoboTwin \
  --manifest benchmark_data/robotwin2/lighting_train_manifest.json \
  --output-root benchmark_data/robotwin2/paired_lighting_replay \
  --finalize-only
```

This pipeline runs clean and randomized-light scenes with identical seeds,
copies complete actor/articulation state at each capture time, and writes paired
RGB/RAW trajectories. It requires the RAW camera additions in the pinned
`third_party/RoboTwin` submodule. For a full dataset, use
`--expected-entries 633` when generating the manifest to enforce the original
episode count.

## Output contract

Keep the generated `manifest.json` beside every replay root. Training configs
must pin the dataset version and must not silently combine different replay
schemas. Validate at least these fields before training:

- trajectory ID and task identity;
- observation/action lengths;
- RAW dtype/range (`float32`, `[0,1]`);
- camera order;
- `burst_frames`, `rnn_frames`, and observation stride;
- state/action normalization statistics.
