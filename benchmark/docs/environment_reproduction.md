# Environment reproduction record

The supported installation flow is documented in
[`ENVIRONMENTS.md`](../../ENVIRONMENTS.md). This file records what was actually
observed on the validation host on 2026-10-02.

## Validated core environments

| Environment | Python | Key packages | Result |
|---|---:|---|---|
| StarVLA | 3.10.20 | torch 2.6.0+cu124, torchvision 0.21.0+cu124, transformers 4.57.0 | RAW-VLA, Dark-ISP, RAM, RAW-Adapter, and RAWild smoke tests passed |
| RoboTwin | 3.10.20 | torch 2.6.0+cu124, SAPIEN 3.0.0b1, MPLib 0.2.1, CuRobo | critical imports, CuRobo planner JIT, and both replay CLI entrypoints passed |
| FastWAM | 3.10.20 | torch 2.7.1+cu128, torchvision 0.22.1+cu128, FastWAM 0.1.0 | CUDA and package imports passed |

The validation host exposed one NVIDIA L4 GPU. PyTorch reported CUDA available
in all three environments above.

A 64×64 headless LIBERO environment reset/render passed with EGL using the
installed LIBERO assets. SAPIEN renderer construction passed in the RoboTwin
environment. The FastWAM RAW camera conversion passed a deterministic uint8 →
RAW10 → uint8 shape/dtype smoke test.

The StarVLA RAW frontend registry also completed a two-timestep/two-burst
RAW-VLA forward and backward pass; 78 parameter tensors received gradients.
The public training YAML resolved successfully through OmegaConf, and the
StarVLA training entrypoint accepted `--help` in the same environment.

## End-to-end replay validation

The official demonstrations already present on the validation host were used
for real simulator replays, not only CLI or import checks:

- LIBERO `libero_spatial`, one 98-step episode: state replay completed and
  produced one RAW NPZ plus one paired RGB NPZ. Both files contained aligned
  `(98, 256, 256, 3)` agent-view and wrist-view arrays, `(98, 7)` actions, and
  `(98, 92)` simulator states.
- RoboTwin `adjust_bottle`, episode 0: closed-loop physics replay succeeded for
  all 140 frames and all three cameras. Frame counts matched the original data;
  aggregate RGB/default-ISP fidelity was 29.868 dB PSNR and 0.985895 SSIM.

Generated validation outputs were written outside the release checkout under
the host's artifact tree and are not part of Git.

## Problems found in pre-existing local environments

The existing StarVLA environment initially did not contain `h5py`, so it could
run the model smoke tests but not the HDF5 simulator replay. It was brought in
line with the public environment by installing `h5py==3.16.0`, after which the
full LIBERO replay passed. A separate LIBERO environment contained `h5py` but
did not contain `bddl`, so it could not construct a LIBERO environment. The
public `environment-starvla-libero.yml` includes both dependencies.

The existing RoboTwin source tree contained CUDA extension binaries built
against a different PyTorch ABI. With the Conda environment's `bin` directory
on `PATH`, CuRobo successfully JIT-compiled fresh extensions into an external
cache and the planner module imported. This is why the environment file pins
`ninja` and the setup instructions require activating the environment.

The pre-existing OpenVLA regeneration environment was not a valid standalone
OpenVLA-OFT environment: it had CPU-only torch 2.1.2 and was missing
torchvision, Transformers, and Hugging Face dependencies. It must not be used
as release evidence.

The available Python 3.12 LeRobot environment did not contain the `gr00t`
package, and the GR00T submodule was not initialized in this staging checkout.
GR00T therefore still requires validation after a recursive clone and
`uv sync --frozen`.

All top-level environment YAML files passed local YAML parsing. A remote Conda
solver dry-run was inconclusive because repodata retrieval did not complete;
no existing environment was modified during validation.

## Snapshot files

The files in `benchmark/docs/env_repro/` preserve Conda and pip package lists
for the working StarVLA and RoboTwin environments. Prefer the reviewed top-level
YAML files for installation and use these snapshots to investigate version
drift.
