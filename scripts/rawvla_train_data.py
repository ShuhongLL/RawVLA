"""Canonical RAWVLA-LIBERO train replay roots.

The older unversioned replay lost task assets for some demos because it did not
reset from each HDF5 demo's ``model_file`` XML. All training config builders
must use the action-XML-fixed pair below.
"""

from __future__ import annotations

import json
from pathlib import Path


RAWVLA_TRAIN_DATA_VERSION = "action_xmlfix_20260816T0839Z"
RAW_DIRNAME = f"rawvla_light_train_raw_npz_{RAWVLA_TRAIN_DATA_VERSION}"
RGB_DIRNAME = f"rawvla_light_train_rgb_replay_npz_{RAWVLA_TRAIN_DATA_VERSION}"
EXPECTED_EPISODES = 1771


def rawvla_train_roots(asset_root: Path) -> tuple[Path, Path]:
    base = Path(asset_root).resolve() / "benchmark_data/libero"
    raw_root = base / RAW_DIRNAME
    rgb_root = base / RGB_DIRNAME
    for root in (raw_root, rgb_root):
        if not root.is_dir():
            raise FileNotFoundError(f"Missing canonical RAWVLA train replay: {root}")

    raw_paths = sorted(path.relative_to(raw_root) for path in raw_root.glob("**/*.npz"))
    rgb_paths = sorted(path.relative_to(rgb_root) for path in rgb_root.glob("**/*.npz"))
    if len(raw_paths) != EXPECTED_EPISODES or len(rgb_paths) != EXPECTED_EPISODES:
        raise RuntimeError(
            "Canonical RAWVLA train replay has an unexpected episode count: "
            f"raw={len(raw_paths)}, rgb={len(rgb_paths)}, expected={EXPECTED_EPISODES}"
        )
    if raw_paths != rgb_paths:
        raw_only = sorted(set(raw_paths) - set(rgb_paths))[:3]
        rgb_only = sorted(set(rgb_paths) - set(raw_paths))[:3]
        raise RuntimeError(
            "Canonical RAWVLA RAW/RGB replay paths are not paired: "
            f"raw_only={raw_only}, rgb_only={rgb_only}"
        )

    import numpy as np

    for root, relative_path, representation in (
        (raw_root, raw_paths[0], "raw"),
        (rgb_root, rgb_paths[0], "rgb_uint8"),
    ):
        with np.load(root / relative_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        if (
            "lighting_sample_unit" not in metadata
            or "demo_key" not in metadata
            or metadata.get("representation") != representation
        ):
            raise RuntimeError(
                f"Replay at {root} does not satisfy the "
                f"{RAWVLA_TRAIN_DATA_VERSION} metadata contract"
            )
    return raw_root, rgb_root
