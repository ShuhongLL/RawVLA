# RoboTwin 2.0 local setup

This workspace uses the RoboTwin 2.0 benchmark at `third_party/RoboTwin` and
the StarVLA `Qwen3-VL-OFT-RoboTwin2-All` checkpoint. The benchmark checkout is
tracked as a submodule on the `submodules/RoboTwin` branch, including the
StarVLA checkpoint forwarding patch required by `script/eval_policy.py`.

Local paths and Python executables are exported by the project `.bashrc`:

```bash
source ./.bashrc
robotwin-activate
```

The required benchmark assets are installed under `third_party/RoboTwin/assets`:
`background_texture`, `embodiments`, and `objects`.

The local environment is `.conda-envs/robotwin2`. It uses Python 3.10,
CUDA 13 nightly PyTorch for B300 support, SAPIEN 3.0.0b1, MPlib 0.2.1, and
CuRobo 0.7.8. The standard RoboTwin SAPIEN and MPlib compatibility patches
are applied inside that local environment. Setuptools remains at 77 or newer
because CUDA 13 nightly PyTorch requires it; CuRobo is installed with build
isolation disabled.

To start a future evaluation after the current LIBERO campaign finishes:

```bash
scripts/robotwin2_eval.sh demo_clean all
scripts/robotwin2_eval.sh demo_randomized all
```

The setup process intentionally does not run a simulator or policy smoke test.
