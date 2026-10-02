# RoboTwin 2.0 clean replay success set

This directory records the reproducible success-only subset from the 13-task,
50-episode-per-task RoboTwin 2.0 clean closed-loop expert replay.  Use this
subset as the input population for later lighting or ISP modifications.

## Files

- `successful_episodes.json`: compact task-to-episode-ID allowlist committed as
  the canonical success set.
- `build_success_manifest.py`: regenerates all success manifests from replay
  records using the strict selection rule below.

The detailed CSV/JSONL exports and machine-specific environment snapshot are
generated locally and intentionally not versioned.

## Selection rule

An episode is included only when all conditions hold:

1. replay record status is `completed`;
2. RoboTwin task result is `success=true`;
3. replay HDF5 exists;
4. PSNR and SSIM are present;
5. original/replay frame counts match for head, left, and right cameras.

The resulting allowlist contains **633 successful episodes across 13 tasks**.
No task-failure episode is present in any generated manifest.

## Image path for lighting work

The baseline comparison is original training `rgb` versus replay
`default_isp`.  Metric computation and the downstream VLA path use
`float32 [0,1]`; `uint8` is allowed only for display/export.  RAW sensor data
is stored separately as RAW10 in `uint16 [0,1023]`.

For a later lighting run, read `successful_episodes.json` as the allowlist and
use the detailed manifest's `source_hdf5`, `source_trajectory`, and
`replay_hdf5` fields.  Do not infer successes again from task indices.

Regenerate with:

```bash
python replay/robotwin2.0/build_success_manifest.py \
  /path/to/robotwin2-clean-replay-result-root
```
