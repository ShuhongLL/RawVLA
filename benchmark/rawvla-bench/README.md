# RAWVLA-Bench-Light LIBERO

This folder contains the standalone LIBERO part of the `rawvla-bench-light`
setting. It implements the lighting setting from
`benchmark/docs/RawVLABench.md`: paired LIBERO rollouts under five
simulator-side lighting domains, followed by the fixed LIBERO RGB-to-pseudo-RAW
unprocess path and a fixed RAW sensor-noise model.

The benchmark is not the previous ISP perturbation setting. The simulator
lighting is changed before rendering; the fixed unprocess/default-ISP path is
then applied to the rendered RGB observation.

The released manifest name is `rawvla-bench-light-libero-v1`.

## Files

- `rawvla_bench/manifest.py`: builds the frozen 40 task x 50 init state x 5
  lighting-domain manifest.
- `rawvla_bench/lighting.py`: stores the five calibrated lighting-domain
  configs and applies `2 ** lighting_ev` to the current MuJoCo model's original
  light and headlight intensities. The real RAWVLA agentview key/table-flood
  light slots are injected into a RAW copy of the current episode's compiled
  XML after the normal LIBERO reset. RAW episodes then reset from that RAW XML;
  later lighting changes only update light intensity arrays.
- `rawvla_bench/raw.py`: imports the current LIBERO
  `raw_rgb10_unprocessing.py`, adds fixed shot/read RAW noise, and returns
  either a RAW view or default-ISP RGB.
- `scripts/build_manifest.py`: writes
  `libero_manifest_50_init_states_10000_rollouts.json` by default.
- `scripts/render_lighting_preview.py`: renders five-domain preview images.
  It also writes `*_raw_normalized_sheet.png`, where each RAW preview is
  independently scaled to the same p95 display luma for visual noise inspection
  only. These normalized PNGs are not benchmark observations. Manifest JSON
  files are the source of truth for test EV/flood values; preview artifacts are
  only visual caches and should be regenerated after lighting config changes.

## Lighting Configs

The current LIBERO v1 lighting domains are defined in
`rawvla_bench/lighting.py` as `LIGHTING_DOMAIN_CONFIGS`. Manifest construction
follows `benchmark/docs/RawVLABench.md`: for each
`(suite, task_id, base_seed, lighting_domain)`, one value is sampled once from
`Uniform(ev_min, ev_max)` with the fixed benchmark seed and then frozen in
the manifest file.

| Domain | EV range | Rig | Agentview table flood | RAW full-well pressure |
|---|---:|---|---:|---:|
| ExtremeLow | `[-5.5, -4.0]` | `common_agentview_key` | `0.0` | `0.0` |
| Low | `[-4.0, -2.5]` | `common_agentview_key` | `0.0` | `0.0` |
| Normal | `[-0.5, +0.5]` | `common_agentview_key` | `0.0` | `0.0` |
| Over | `[+1.5, +3.0]` | `common_agentview_key` | `[0.00, 1.00]` | `[0.50, 1.00]` |
| ExtremeOver | `[+6.0, +8.0]` | `common_agentview_key` | `[1.00, 2.00]` | `[1.00, 1.50]` |

The extra RAW full-well pressure is a continuous term applied before
noise/default ISP. For domains with flood and saturation ranges, the same
per-entry uniform sample `u` is used for EV, flood, and RAW saturation, so
brighter EV samples also receive stronger table flood and stronger full-well
pressure within the same domain.

The `common_agentview_key` rig is a real MuJoCo XML light topology, not a
headlight approximation. For each episode, RAWVLA-Bench first runs the normal
LIBERO reset with the manifest seed, then derives a RAW XML from that compiled
scene by adding the extra light slots. This matters because some suites encode
fixed fixtures / body placements in XML rather than qpos. RAW episodes reset
from this episode-specific RAW XML, while RGB baseline episodes keep the
original reset scene. Runtime lighting application must not generate or switch
XML; it only applies the frozen manifest entry:

```text
original LIBERO lights = captured_original_lights * lighting_scale
headlight = captured_original_headlight * lighting_scale
rawvla_agentview_key = 0.02 * lighting_scale
rawvla_agentview_flood_* = agentview_table_flood
```

`agentview_table_flood` is the manifest/config value directly; it is not
multiplied again by EV.

## Paired Episodes

The default LIBERO test manifest is
`libero_manifest_50_init_states_10000_rollouts.json`. It uses all 50 fixed
LIBERO initial states per task and evaluates each one under the five lighting
domains, for 250 rollouts per task:

```text
50 init_state_index groups x 5 lighting domains = 250 episodes/task
```

Across all four standard LIBERO suites, this default protocol has:

```text
40 tasks x 50 init states x 5 lighting domains = 10000 rollouts
```

For quicker compatibility checks, the smaller
`libero_manifest_10_init_states_2000_rollouts.json` keeps the first 10 fixed
initial states per task:

```text
40 tasks x 10 init states x 5 lighting domains = 2000 rollouts
```

Within one `base_seed_id` group, the five lighting-domain entries share the
same task, language instruction, camera setup, `base_seed`, and
`init_state_index`. Therefore object/robot initialization is paired across
`ExtremeLow`, `Low`, `Normal`, `Over`, and `ExtremeOver`.

The fields that intentionally change across the five paired entries are the
sampled lighting fields (`lighting_domain`, `lighting_ev`, `lighting_scale`,
and derived light rig parameters such as `agentview_table_flood`) plus the
deterministic `noise_seed`, which is frozen per
`(suite, task_id, base_seed, lighting_domain)`.

## Build Manifest

```bash
PYTHONPATH=benchmark/rawvla-bench \
python benchmark/rawvla-bench/scripts/build_manifest.py
```

## Evaluate With StarVLA LIBERO

Use the patched `eval_libero.py` with:

```bash
export RAWVLA_ROOT="$(pwd)"
--args.image-mode rawvla_bench \
--args.rawvla-bench-manifest $RAWVLA_ROOT/benchmark/rawvla-bench/libero_manifest_50_init_states_10000_rollouts.json \
--args.rawvla-bench-representation raw
```

For the default ISP observation, set:

```bash
--args.rawvla-bench-representation default_isp
```

The evaluation ledger records the lighting domain, EV, base seed, initial-state
index, and noise seed for each completed rollout.

## Evaluate Pluggable RAW Frontends

`scripts/run_libero_models.sh` can compose each of the six LIBERO backbones with
one independently trained RAW frontend. The supported values are:

| `RAW_FRONTEND` | Behavior |
|---|---|
| `none` | No ISP/frontend module; the selected benchmark representation is sent directly to the backbone. |
| `identity` | An explicit parameter-free identity frontend, useful for checking the frontend code path. |
| `rawvla` | RAWVLA, using the per-backbone 2k checkpoint by default. |
| `darkisp` | The per-backbone DarkISP checkpoint. |
| `ram` | The per-backbone RAW Adaptation Module checkpoint. |
| `raw_adapter` | The per-backbone RAW-Adapter checkpoint. |
| `rawild` | The per-backbone RAWild checkpoint. |

For example, evaluate RAWVLA on RAW observations across all six backbones:

```bash
RAW_FRONTEND=rawvla \
RAWVLA_CHECKPOINT_STEP=2000 \
bash benchmark/rawvla-bench/scripts/run_libero_bench.sh
```

Evaluate without an ISP module:

```bash
RAW_FRONTEND=none \
RAWVLA_BENCH_REPRESENTATION=raw \
bash benchmark/rawvla-bench/scripts/run_libero_bench.sh
```

Replace `rawvla` with `darkisp`, `ram`, `raw_adapter`, or `rawild` to select a
baseline. Runs are isolated under `<LOG_ROOT>/<frontend>/<model>/<suite>` so
resume ledgers from different frontend conditions cannot collide. Use
`RAW_FRONTEND_CHECKPOINT_ROOT` to relocate the `baselines` directory.

The resolved six-backbone composition can be checked without loading a model
or launching an evaluator:

```bash
RAW_FRONTEND=rawvla \
IMAGE_MODE=rawvla_bench \
RAWVLA_BENCH_REPRESENTATION=raw \
RAW_FRONTEND_EVAL_DRY_RUN=1 \
bash benchmark/rawvla-bench/scripts/run_libero_models.sh
```
