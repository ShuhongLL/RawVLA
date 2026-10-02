# Replay pipelines

Replay outputs are intentionally not stored in Git. By default they live under
`benchmark_data/`, which is ignored. Every command below supports explicit
input/output paths and should first be run with one episode or one worker.

Install the StarVLA/LIBERO environment for LIBERO replay and the RoboTwin
environment for RoboTwin replay by following [`ENVIRONMENTS.md`](../ENVIRONMENTS.md).
The LIBERO HDF5 pipeline requires both `h5py` and `bddl`; run the environment
checker before starting a long replay.

## Download source demonstrations

Install the Hugging Face CLI and run:

```bash
bash scripts/download_replay_datasets.sh
```

Use `DATA_ROOT=/path/to/benchmark_data` to place data outside the checkout.
`PROXY_HTTP` and `PROXY_ALL` are optional; no proxy is enabled by default.

## LIBERO HDF5 simulator replay

This is the canonical paired replay: it restores each demonstration's simulator
state/XML, applies RAWVLA-Bench lighting, and writes target-light RAW plus paired
default-light RGB NPZ files.

```bash
python scripts/replay_libero_train_rawvla_light_npz.py \
  --hdf5-root benchmark_data/libero/LIBERO-datasets \
  --out-root benchmark_data/libero/rawvla_light_train_raw_npz \
  --rgb-out-root benchmark_data/libero/rawvla_light_train_rgb_replay_npz \
  --max-episodes 1 --workers 1
```

Remove `--max-episodes 1` only after inspecting the generated manifest and NPZ
schema. `--replay-mode states` is the reproducible default.

## LIBERO RLDS conversion

For existing modified LIBERO RLDS shards:

```bash
python scripts/build_rawvla_light_train_npz_cache.py \
  --data-root benchmark_data/libero/modified_libero_rlds \
  --out-root benchmark_data/libero/rawvla_light_train_cache_npz \
  --max-episodes 1 --workers 1
```

## RoboTwin 2.0 clean expert replay

```bash
python scripts/replay_robotwin2_clean_training.py --help
```

The script replays saved joint trajectories through RoboTwin physics and checks
render fidelity against the original training frames. Inputs are explicit CLI
arguments; start with one selected episode.

## RoboTwin 2.0 paired-lighting replay

```bash
python scripts/replay_robotwin2_paired_lighting_training.py --help
```

This pipeline runs clean and randomized-light scenes with identical seeds,
copies complete actor/articulation state at each capture time, and writes paired
RGB/RAW trajectories. It requires the RAW camera additions in the pinned
`third_party/RoboTwin` submodule.

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
