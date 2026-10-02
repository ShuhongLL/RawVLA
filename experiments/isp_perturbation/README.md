# ISP Perturbation Experiments

This directory contains the earlier synthetic LIBERO ISP-perturbation studies:
exposure-value transforms, bit-depth reduction, chromatic shifts, and tonal
response changes. These experiments operate on rendered observations and are
kept separate from RAWVLA-Bench.

RAWVLA-Bench changes simulator lighting before rendering and then applies its
fixed RGB-to-pseudo-RAW and sensor-noise path. Its implementation, frozen
manifests, and official runners live in
[`benchmark/rawvla-bench`](../../benchmark/rawvla-bench/README.md).

## Layout

- `configs/`: perturbation grids and named profiles.
- `scripts/`: experiment launchers and result summarizers.
- `results/`: checked-in aggregate reports; generated ledgers remain outside
  version control by default.

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

These launchers require the StarVLA/LIBERO environment and downloaded policy
checkpoints described in the root `ENVIRONMENTS.md`.
