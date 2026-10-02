"""Direct trajectory-window loader for RAWVLA replay NPZ files."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def collate_fn(batch):
    return batch


class RAWVLALightNPZDataset(Dataset):
    """Sample causal policy-observation windows without crossing episodes.

    Two temporal axes are kept separate: each recurrent observation is
    separated by ``observation_stride`` env steps (the policy replanning
    cadence), while its denoising burst contains consecutive camera frames.
    A small episode-local cache amortizes loading the trajectory arrays.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.cache_root = Path(str(cfg.cache_root))
        rgb_cache_root = str(cfg.get("rgb_cache_root", "") or "")
        self.rgb_cache_root = Path(rgb_cache_root) if rgb_cache_root else None
        self.dataset_format = str(cfg.get("dataset_format", "libero")).lower()
        self.dataset_version = str(cfg.get("dataset_version", ""))
        if self.dataset_version and self.dataset_version not in self.cache_root.name:
            raise ValueError(
                f"RAWVLA dataset_version={self.dataset_version!r} does not match "
                f"cache_root={self.cache_root}"
            )
        self.paths = sorted(glob.glob(str(self.cache_root / "**" / "*.npz"), recursive=True))
        if not self.paths:
            raise FileNotFoundError(f"No RAWVLA replay NPZ files under {self.cache_root}")
        domain_sampling_weights = dict(cfg.get("lighting_domain_sampling_weights", {}) or {})
        self.path_sampling_probabilities = None
        self.path_lighting_domains = None
        if domain_sampling_weights:
            normalized_weights = {
                str(domain).lower(): float(weight)
                for domain, weight in domain_sampling_weights.items()
            }
            if any(weight <= 0.0 for weight in normalized_weights.values()):
                raise ValueError("lighting-domain sampling weights must be positive")
            path_weights = []
            path_domains = []
            for path in self.paths:
                with np.load(path, allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata_json"].item()))
                domain = str(metadata.get("lighting_domain", "")).lower()
                path_domains.append(domain)
                path_weights.append(normalized_weights.get(domain, 1.0))
            probabilities = np.asarray(path_weights, dtype=np.float64)
            self.path_sampling_probabilities = probabilities / probabilities.sum()
            self.path_lighting_domains = path_domains
        self.burst_frames = int(cfg.get("burst_frames", 6))
        self.rnn_frames = int(cfg.get("rnn_frames", 6))
        self.observation_stride = int(cfg.get("observation_stride", cfg.get("burst_stride", 1)))
        self.action_horizon = int(cfg.get("action_horizon", 8))
        self.action_offset = int(cfg.get("action_offset", 0))
        self.state_source_key = str(cfg.get("state_source_key", "robot_states"))
        self.samples_per_episode = int(cfg.get("samples_per_episode", 32))
        self.virtual_length = int(cfg.get("virtual_length", 100000))
        self.seed = int(cfg.get("seed", 42))
        self.normalization = str(cfg.get("normalization", "starvla_minmax"))
        self.stats_path = Path(str(cfg.get("stats_path", "")))
        self.stats_key = str(cfg.get("stats_key", ""))
        self.action_stats_key = str(cfg.get("action_stats_key", "actions"))
        self.state_stats_key = str(cfg.get("state_stats_key", "state"))
        self.delta_action_indices = tuple(int(index) for index in cfg.get("delta_action_indices", ()))
        self._cached_path = None
        self._cached = None
        self._cached_rgb_path = None
        self._cached_rgb = None
        self._stats = self._load_stats()
        if min(self.burst_frames, self.rnn_frames, self.observation_stride, self.action_horizon) < 1:
            raise ValueError(
                "burst_frames, rnn_frames, observation_stride and action_horizon must be positive"
            )
        if self.action_offset < 0:
            raise ValueError("action_offset must be non-negative")

    def _load_stats(self):
        if not self.stats_path.is_file():
            raise FileNotFoundError(f"Normalization statistics not found: {self.stats_path}")
        with open(self.stats_path) as handle:
            data = json.load(handle)
        if self.stats_key:
            for component in self.stats_key.split("."):
                data = data[component]
            return data
        if self.normalization == "starvla_minmax":
            if self.dataset_format == "robotwin2":
                return data["new_embodiment"]
            return data["franka"]
        return data.get("norm_stats", data)

    def __len__(self):
        return self.virtual_length

    def set_epoch(self, epoch: int):
        return None

    def save_dataset_statistics(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(
                {
                    "source": str(self.cache_root),
                    "dataset_version": self.dataset_version,
                    "episodes": len(self.paths),
                    "normalization": self.normalization,
                    "normalization_stats": str(self.stats_path),
                    "burst_frames": self.burst_frames,
                    "rnn_frames": self.rnn_frames,
                    "observation_stride": self.observation_stride,
                },
                handle,
                indent=2,
            )

    def _load_episode(self, path: str):
        if path != self._cached_path:
            with np.load(path, allow_pickle=False) as archive:
                self._cached = {key: np.asarray(archive[key]) for key in archive.files}
            self._cached_path = path
        return self._cached

    def _load_rgb_episode(self, raw_path: str):
        if self.rgb_cache_root is None:
            return None
        relative = Path(raw_path).relative_to(self.cache_root)
        rgb_path = self.rgb_cache_root / relative
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing paired RGB chroma target: {rgb_path}")
        rgb_path_string = str(rgb_path)
        if rgb_path_string != self._cached_rgb_path:
            with np.load(rgb_path, allow_pickle=False) as archive:
                self._cached_rgb = {key: np.asarray(archive[key]) for key in archive.files}
            self._cached_rgb_path = rgb_path_string
        return self._cached_rgb

    @staticmethod
    def _minmax(value, low, high):
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        return np.clip(2.0 * (value - low) / np.maximum(high - low, 1.0e-6) - 1.0, -1.0, 1.0)

    def _normalize(self, actions, state):
        actions = actions.astype(np.float32, copy=True)
        state = state.astype(np.float32, copy=True)
        if self.normalization == "starvla_minmax":
            if self.dataset_format == "robotwin2":
                # RoboTwin env/replay order is
                # [left arm, left gripper, right arm, right gripper], while
                # StarVLA's AgileX training order places both arms first.
                order = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13])
                actions = actions[:, order]
                stats = self._stats["action"]
                actions[:, :12] = self._minmax(
                    actions[:, :12], stats["min"][:12], stats["max"][:12]
                )
                actions[:, 12:] = (actions[:, 12:] > 0.49).astype(np.float32)
                return actions, None
            stats = self._stats["action"]
            actions[:, :6] = self._minmax(actions[:, :6], stats["min"][:6], stats["max"][:6])
            actions[:, 6] = (actions[:, 6] + 1.0) * 0.5
            return actions, None
        if self.normalization in {"zscore", "pi0_zscore"}:
            # Delta conversion is an embodiment-level policy setting.  It is
            # expressed explicitly as indices instead of being tied to LIBERO
            # or RoboTwin action layouts.
            delta_indices = self.delta_action_indices
            if self.normalization == "pi0_zscore" and not delta_indices:
                delta_indices = tuple(range(min(6, actions.shape[-1], state.shape[-1])))
            if delta_indices:
                indices = np.asarray(delta_indices, dtype=np.int64)
                actions[:, indices] -= state[None, indices]
            action_stats = self._stats[self.action_stats_key]
            state_stats = self._stats[self.state_stats_key]
            action_dim = actions.shape[-1]
            state_dim = state.shape[-1]
            actions = (actions - np.asarray(action_stats["mean"][:action_dim])) / np.maximum(
                np.asarray(action_stats["std"][:action_dim]), 1.0e-6
            )
            state = (state - np.asarray(state_stats["mean"][:state_dim])) / np.maximum(
                np.asarray(state_stats["std"][:state_dim]), 1.0e-6
            )
            return actions.astype(np.float32), state.astype(np.float32)
        if self.normalization in {"quantile", "pi05_quantile"}:
            action_stats = self._stats[self.action_stats_key]
            state_stats = self._stats[self.state_stats_key]
            action_dim = actions.shape[-1]
            state_dim = state.shape[-1]
            actions = self._minmax(actions, action_stats["q01"][:action_dim], action_stats["q99"][:action_dim])
            state = self._minmax(state, state_stats["q01"][:state_dim], state_stats["q99"][:state_dim])
            return actions.astype(np.float32), state.astype(np.float32)
        raise ValueError(f"Unknown normalization mode: {self.normalization}")

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index) * 104729)
        if self.path_sampling_probabilities is None:
            episode_slot = (int(index) // self.samples_per_episode) % len(self.paths)
        else:
            episode_block = int(index) // self.samples_per_episode
            path_rng = np.random.default_rng(self.seed + episode_block * 104729)
            episode_slot = int(
                path_rng.choice(len(self.paths), p=self.path_sampling_probabilities)
            )
        path = self.paths[episode_slot]
        episode = self._load_episode(path)
        rgb_episode = self._load_rgb_episode(path)
        num_frames = int(episode["actions"].shape[0])
        current = int(rng.integers(0, num_frames))

        observation_indices = current - self.observation_stride * np.arange(
            self.rnn_frames - 1, -1, -1
        )
        observation_indices = np.clip(observation_indices, 0, num_frames - 1)
        # Inner K axis is always adjacent camera frames for burst denoising.
        burst_offsets = np.arange(self.burst_frames - 1, -1, -1)
        burst_indices = np.clip(observation_indices[:, None] - burst_offsets[None, :], 0, num_frames - 1)
        action_start = current + self.action_offset
        action_indices = np.clip(
            np.arange(action_start, action_start + self.action_horizon), 0, num_frames - 1
        )
        if self.state_source_key not in episode:
            raise KeyError(f"state_source_key={self.state_source_key!r} missing from {path}")
        state = episode[self.state_source_key][current].astype(np.float32)
        actions, normalized_state = self._normalize(episode["actions"][action_indices], state)
        metadata = json.loads(str(episode["metadata_json"].item()))

        bursts = []
        layout_images = []
        chroma_targets = []
        if self.dataset_format == "robotwin2":
            raw_keys = (
                "head_camera_raw_float32",
                "left_camera_raw_float32",
                "right_camera_raw_float32",
            )
            rgb_keys = (
                "head_camera_rgb_uint8",
                "left_camera_rgb_uint8",
                "right_camera_rgb_uint8",
            )
        elif self.dataset_format == "libero":
            raw_keys = ("agentview_raw_uint8", "wrist_raw_uint8")
            rgb_keys = raw_keys
        else:
            raise ValueError(f"Unknown RAWVLA dataset_format: {self.dataset_format}")

        for raw_key, rgb_key in zip(raw_keys, rgb_keys):
            # [T_rnn, K_burst, H, W, C]
            burst = episode[raw_key][burst_indices].astype(np.float32)
            if episode[raw_key].dtype == np.uint8:
                burst /= 255.0
            bursts.append(np.ascontiguousarray(burst))
            if rgb_episode is not None and rgb_key in rgb_episode:
                target = np.asarray(rgb_episode[rgb_key][current], dtype=np.float32) / 255.0
                chroma_targets.append(np.ascontiguousarray(target))
                layout = rgb_episode[rgb_key][current]
            elif rgb_key in episode:
                layout = episode[rgb_key][current]
            else:
                layout = np.rint(np.clip(episode[raw_key][current], 0.0, 1.0) * 255.0).astype(
                    np.uint8
                )
            layout_images.append(Image.fromarray(layout, mode="RGB"))
        language = str(metadata.get("task_description", ""))
        if not language and self.dataset_format == "robotwin2":
            language = str(metadata.get("task_name", "")).replace("_", " ")
        sample = {
            "image": layout_images,
            "raw_burst_float32": bursts,
            "action": actions,
            "lang": language,
            "rawvla_light": metadata,
        }
        if normalized_state is not None:
            sample["state"] = normalized_state[None]
        if chroma_targets:
            if len(chroma_targets) != len(bursts):
                raise RuntimeError("Paired RGB chroma targets are incomplete for this sample")
            sample["rawvla_chroma_target_float32"] = chroma_targets
        return sample


def get_vla_dataset(data_cfg, **_):
    return RAWVLALightNPZDataset(data_cfg)
