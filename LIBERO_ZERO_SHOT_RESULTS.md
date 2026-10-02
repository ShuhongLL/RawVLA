# LIBERO RGB Zero-shot Evaluation

> 输入为原始 LIBERO RGB 双视角（`agentview` + `robot0_eye_in_hand`）。每个 suite
> 包含 10 个任务，每个任务使用 50 个固定初始状态，共 500 episodes。所有结果均已完成。

最后更新：2026-07-23 11:42:27 UTC

| Model | Spatial | Object | Goal | Long (LIBERO-10) | 4-suite Avg |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-FAST-LIBERO-4in1 | 90.60% | 98.00% | 91.80% | 86.80% | 91.80% |
| Qwen2.5-VL-GR00T-LIBERO-4in1 | 97.60% | 99.00% | 96.60% | 91.60% | 96.20% |
| Qwen2.5-VL-OFT-LIBERO-4in1 | 98.00% | 97.60% | 97.40% | 88.00% | 95.25% |
| Qwen3-VL-OFT-LIBERO-4in1 | 98.60% | 100.00% | 98.40% | 94.00% | **97.75%** |
| Qwen3-VL-PI-LIBERO-4in1 | 98.40% | 99.00% | 98.00% | 95.20% | 97.65% |
| WM4A-CosmoPredict-GR00T-LIBERO-4in1 | 96.00% | 98.60% | 93.20% | 76.00% | 90.95% |
| WM4A-Wan2d2-OFT-LIBERO-4in1 | 96.40% | 97.80% | 93.20% | 87.60% | 93.75% |
| PI0-LIBERO | 96.60% | 97.80% | 93.00% | 82.00% | 92.35% |
| PI0.5-LIBERO | 98.00% | 99.40% | 97.60% | 93.20% | 97.05% |

## Aggregate

- Checkpoints: 9
- Suites per checkpoint: 4
- Episodes per suite: 500
- Total episodes: 18,000
- Total successes: 17,055
- Micro-average success rate: **94.75%**
- Best 4-suite average: **Qwen3-VL-OFT-LIBERO-4in1 — 97.75%**

## Result provenance

- Primary RGB run: `log/libero_zeroshot/`
- Repair RGB run: `log/libero_zeroshot_repair/`
- Qwen3-VL-PI, PI0 and PI0.5 use the completed repair results.
- WM4A-Wan2d2-OFT uses the repair result for `libero_goal`; its other suites use the primary run.
- All remaining entries use the primary run.
- A result is included only when its suite contains all 500 episodes and the job has a completion marker.

## Run metadata

- Seed: 7
- Trials per task: 50
- Tasks per suite: 10
- Camera views: 2 RGB images
- Videos: disabled for throughput
- Evaluation suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
