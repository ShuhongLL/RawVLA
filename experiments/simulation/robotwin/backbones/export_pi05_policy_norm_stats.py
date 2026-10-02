#!/usr/bin/env python3
"""Export the RoboTwin pi0.5 policy normalizer to the RAWVLA loader JSON schema."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

FIELDS = ("mean", "std", "min", "max", "q01", "q99")
SOURCE_KEYS = {
    "actions": "action",
    "state": "observation.state",
}
EXPECTED_DIMENSION = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export policy state/action statistics from a LeRobot pi0.5 checkpoint."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing the downloaded SidneyXie/pi05_robotwin checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON. Defaults to <checkpoint-dir>/policy_norm_stats.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from safetensors import safe_open

    checkpoint_dir = args.checkpoint_dir.resolve()
    source_path = checkpoint_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    output_path = (args.output or checkpoint_dir / "policy_norm_stats.json").resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Policy normalizer not found: {source_path}")

    payload: dict[str, dict[str, list[float]]] = {key: {} for key in SOURCE_KEYS}
    with safe_open(str(source_path), framework="numpy") as source:
        available = set(source.keys())
        for output_key, source_key in SOURCE_KEYS.items():
            for field in FIELDS:
                tensor_key = f"{source_key}.{field}"
                if tensor_key in available:
                    values = source.get_tensor(tensor_key).reshape(-1).astype(float).tolist()
                    payload[output_key][field] = values

    for feature, statistics in payload.items():
        missing = {"mean", "std"} - statistics.keys()
        if missing:
            raise KeyError(f"Missing {sorted(missing)} for {feature} in {source_path}")
        for field in ("mean", "std"):
            values = statistics[field]
            if len(values) != EXPECTED_DIMENSION:
                raise ValueError(
                    f"Expected {EXPECTED_DIMENSION} values for {feature}.{field}, got {len(values)}"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite value in {feature}.{field}")
        if any(value <= 0.0 for value in statistics["std"]):
            raise ValueError(f"Non-positive standard deviation in {feature}.std")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump({"norm_stats": payload}, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(output_path)


if __name__ == "__main__":
    main()
