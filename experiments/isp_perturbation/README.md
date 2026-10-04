# ISP Perturbation Experiments

The synthetic ISP-perturbation studies cover both LIBERO and RoboTwin 2.0,
including the RoboTwin FastWAM baseline. They vary exposure, sensor noise, bit
depth, chromatic response, and tone on observations and are separate from
RawVLA-Bench's simulator-lighting evaluation.

RawVLA-Bench changes simulator lighting before rendering and then applies its
fixed RGB-to-pseudo-RAW and sensor-noise path. Its benchmark integrations live
in [`experiments/simulation/libero`](../simulation/libero/README.md) and
[`experiments/simulation/robotwin`](../simulation/robotwin/README.md).

## Layout

- `configs/`: LIBERO perturbation grids and named profiles.
- `scripts/`: LIBERO experiment launchers and result summarizers.
- `results/`: checked-in aggregate reports; generated ledgers remain outside
  version control by default.

RoboTwin perturbation operators are implemented in the pinned
`third_party/RoboTwin` checkout and in FastWAM's
`third_party/FastWAM/third_party/RoboTwin` checkout. The RoboTwin evaluation
entry points are under `scripts/evaluation/`, not this directory's `scripts/`.

## Environments

| Study | Runtime |
| --- | --- |
| LIBERO with StarVLA policies | `starvla-libero` from `environment-libero.yml` |
| RoboTwin 2.0 with a StarVLA policy | `robotwin` from `environment-robotwin2.yml` for the simulator, plus `starvla-libero` for the policy server |
| RoboTwin 2.0 with FastWAM | `fastwam` from `environment-fastwam.yml`, using FastWAM's pinned RoboTwin checkout |

Install these only for the studies you run; the commands are in the root
README and [`ENVIRONMENTS.md`](../../ENVIRONMENTS.md). Use the corresponding
policy checkpoint and simulator assets for each case.

## LIBERO launchers

All launchers derive the repository root from their own location. Override
machine-specific locations with `RAWVLA_ROOT`, `DATA_ROOT`, `PRETRAINED_ROOT`,
`OPENPI`, and `RESULT_ROOT` rather than editing scripts.

The float32 EV matrix additionally accepts `GPU_IDS`, `JOBS_PER_GPU`,
`LIBERO_MODELS`, `LIBERO_SUITES`, `EV_VALUES`, and `EV_REPRESENTATIONS`; none
of these machine/run dimensions need to be edited in the launcher.

Examples:

```bash
bash experiments/isp_perturbation/scripts/run_bitdepth_setting.sh \
  Qwen3-VL-OFT-LIBERO-4in1 libero_spatial 8 1

bash experiments/isp_perturbation/scripts/run_chromatic_condition.sh \
  Qwen3-VL-OFT-LIBERO-4in1 uv_whitepoint uv_r0p075 90 1
```

These launchers require the `starvla-libero` environment and downloaded policy
checkpoints.

## RoboTwin 2.0 launchers

Both RoboTwin checkouts expose the ISP result as the `isp_perturbed` camera
product. Set **both** `ROBOTWIN_ISP_AXIS` and
`ROBOTWIN_IMAGE_OBS_KEY=isp_perturbed`; otherwise the policy's default image
key bypasses the perturbation. For example, to evaluate one bit-depth setting
with a StarVLA policy:

```bash
ROBOTWIN_ISP_AXIS=bitdepth ROBOTWIN_ISP_BITS=6 \
ROBOTWIN_IMAGE_OBS_KEY=isp_perturbed \
  bash scripts/evaluation/run_robotwin2_starvla_rgb_eval.sh one adjust_bottle
```

For the FastWAM baseline, activate `fastwam` and run the same camera setting
through its separate evaluation entry point:

```bash
ROBOTWIN_ISP_AXIS=bitdepth ROBOTWIN_ISP_BITS=6 \
ROBOTWIN_IMAGE_OBS_KEY=isp_perturbed \
  bash scripts/evaluation/run_fastwam_robotwin2_eval.sh one adjust_bottle
```

The available `ROBOTWIN_ISP_AXIS` values are `ev`, `noise_level`, `bitdepth`,
`chromatic`, and `tonal`. Each requires its corresponding `ROBOTWIN_*` setting
(for example, `ROBOTWIN_ISP_EV` and `ROBOTWIN_ISP_EV_REPRESENTATION` for `ev`).
These are individual evaluation settings, not a dedicated full-sweep launcher;
see the two pinned `envs/camera/raw_rgb10.py` implementations for parameter
names and allowed values.
