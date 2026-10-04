# Scripts

Run these scripts from the repository root. General installation details are
in [ENVIRONMENTS.md](../ENVIRONMENTS.md).

## setup/

- [configure_libero.py](setup/configure_libero.py) writes the ignored,
  machine-local `.local/libero/config.yaml` with simulator and dataset paths.
- [setup_robotwin_assets.sh](setup/setup_robotwin_assets.sh) downloads the
  RoboTwin simulator assets or links an existing asset tree.
- [check_environment.py](setup/check_environment.py) validates the selected
  Python environment and optional simulator rendering.

## replay/

The optional data-regeneration scripts, their environments, inputs, command
order, and outputs are documented in [replay/README.md](../replay/README.md).

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
