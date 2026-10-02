from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def _load_dataset_class():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "starVLA"
        / "dataloader"
        / "rawvla_npz_datasets.py"
    )
    spec = importlib.util.spec_from_file_location("rawvla_npz_datasets_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RAWVLALightNPZDataset


RAWVLALightNPZDataset = _load_dataset_class()


def test_robotwin_paired_npz_contract(tmp_path) -> None:
    raw_root = tmp_path / "paired" / "raw"
    rgb_root = tmp_path / "paired" / "rgb"
    raw_episode = raw_root / "grab_roller" / "episode_000000.npz"
    rgb_episode = rgb_root / "grab_roller" / "episode_000000.npz"
    raw_episode.parent.mkdir(parents=True)
    rgb_episode.parent.mkdir(parents=True)

    frame_count = 12
    height, width = 4, 5
    actions = np.arange(frame_count * 14, dtype=np.float32).reshape(frame_count, 14) / 10.0
    metadata = np.asarray(
        json.dumps(
            {
                "task_name": "grab_roller",
                "lighting_domain": "Low",
            }
        )
    )
    raw_arrays = {
        "actions": actions,
        "metadata_json": metadata,
    }
    rgb_arrays = {"metadata_json": metadata}
    for view_index, view in enumerate(("head_camera", "left_camera", "right_camera")):
        raw_arrays[f"{view}_raw_float32"] = np.full(
            (frame_count, height, width, 3),
            0.1 * (view_index + 1),
            dtype=np.float32,
        )
        rgb_arrays[f"{view}_rgb_uint8"] = np.full(
            (frame_count, height, width, 3),
            10 * (view_index + 1),
            dtype=np.uint8,
        )
    np.savez(raw_episode, **raw_arrays)
    np.savez(rgb_episode, **rgb_arrays)

    stats_path = tmp_path / "policy_norm_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "actions": {"mean": [0.0] * 14, "std": [2.0] * 14},
                    "state": {"mean": [0.0] * 14, "std": [4.0] * 14},
                }
            }
        ),
        encoding="utf-8",
    )

    seed = 7
    dataset = RAWVLALightNPZDataset(
        OmegaConf.create(
            {
                "cache_root": str(raw_root),
                "rgb_cache_root": str(rgb_root),
                "dataset_format": "robotwin2",
                "stats_path": str(stats_path),
                "stats_key": "norm_stats",
                "normalization": "zscore",
                "action_stats_key": "actions",
                "state_stats_key": "state",
                "state_source_key": "actions",
                "delta_action_indices": [],
                "burst_frames": 2,
                "rnn_frames": 3,
                "observation_stride": 2,
                "action_horizon": 4,
                "action_offset": 1,
                "samples_per_episode": 1,
                "virtual_length": 1,
                "seed": seed,
            }
        )
    )

    sample = dataset[0]
    current = int(np.random.default_rng(seed).integers(0, frame_count))
    action_indices = np.clip(np.arange(current + 1, current + 5), 0, frame_count - 1)

    assert len(sample["raw_burst_float32"]) == 3
    assert sample["raw_burst_float32"][0].shape == (3, 2, height, width, 3)
    assert all(image.mode == "RGB" for image in sample["image"])
    assert sample["state"].shape == (1, 14)
    np.testing.assert_allclose(sample["state"][0], actions[current] / 4.0)
    np.testing.assert_allclose(sample["action"], actions[action_indices] / 2.0)
    assert sample["lang"] == "grab roller"
    assert sample["rawvla_light"]["lighting_domain"] == "Low"

    chroma_targets = sample["rawvla_chroma_target_float32"]
    assert len(chroma_targets) == 3
    np.testing.assert_allclose(chroma_targets[0], np.float32(10.0 / 255.0))
    np.testing.assert_allclose(chroma_targets[1], np.float32(20.0 / 255.0))
    np.testing.assert_allclose(chroma_targets[2], np.float32(30.0 / 255.0))
