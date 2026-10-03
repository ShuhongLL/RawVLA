# Scripts

Run these scripts from the repository root. The published paired training data
can be downloaded directly from [RawVLA-Bench on Hugging Face](https://huggingface.co/datasets/ToferFish/RawVLA-Bench);
the replay scripts below are only needed to regenerate it from source
demonstrations. Installation details are in [ENVIRONMENTS.md](../ENVIRONMENTS.md).

## setup/

- [configure_libero.py](setup/configure_libero.py) writes the ignored,
  machine-local `.local/libero/config.yaml` with simulator and dataset paths.
- [setup_robotwin_assets.sh](setup/setup_robotwin_assets.sh) downloads the
  RoboTwin simulator assets or links an existing asset tree.
- [check_environment.py](setup/check_environment.py) validates the selected
  Python environment and optional simulator rendering.

## replay/

- [download_replay_datasets.sh](replay/download_replay_datasets.sh) downloads
  the original LIBERO demonstrations and RoboTwin archives for regeneration;
  it does **not** download the published RawVLA-Bench training pairs.
- [replay_libero_train_rawvla_light_npz.py](replay/replay_libero_train_rawvla_light_npz.py)
  re-renders LIBERO demonstrations into paired RAW/RGB trajectories.
- [replay_robotwin2_clean_training.py](replay/replay_robotwin2_clean_training.py)
  replays RoboTwin expert trajectories and checks the clean render.
- [build_robotwin2_lighting_train_manifest.py](replay/build_robotwin2_lighting_train_manifest.py)
  fixes lighting assignments for successful RoboTwin clean replays.
- [replay_robotwin2_paired_lighting_training.py](replay/replay_robotwin2_paired_lighting_training.py)
  uses that manifest to produce paired RoboTwin RAW/RGB trajectories.

See [replay/README.md](../replay/README.md) for the input data, command order,
and output paths.

## evaluation/

- [robotwin2_eval.sh](evaluation/robotwin2_eval.sh) forwards a checkpoint and
  mode to StarVLA's RoboTwin evaluator.
- [run_robotwin2_starvla_rgb_eval.sh](evaluation/run_robotwin2_starvla_rgb_eval.sh)
  runs the StarVLA RGB RoboTwin checkpoint/task matrix.
- [run_fastwam_robotwin2_eval.sh](evaluation/run_fastwam_robotwin2_eval.sh)
  runs the FastWAM RoboTwin checkpoint/task matrix.
- [summarize_robotwin2_eval.py](evaluation/summarize_robotwin2_eval.py)
  summarizes completion markers from the RoboTwin evaluation runner.

LIBERO evaluation runners live under
[experiments/simulation/libero/scripts/](../experiments/simulation/libero/scripts/).
The frozen RawVLA-Bench evaluation manifests are in the corresponding
[LIBERO](../experiments/simulation/libero/manifests/) and
[RoboTwin](../experiments/simulation/robotwin/manifests/) directories. The
RoboTwin wrappers above do not automatically load their JSON manifest.
