#!/usr/bin/env python3
"""Create the machine-local LIBERO path configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=ROOT / "LIBERO-git" / "libero" / "libero",
        help="Directory containing LIBERO assets, BDDL files, and init files.",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        default=ROOT / "benchmark_data" / "libero" / "LIBERO-datasets",
        help="Directory containing downloaded LIBERO demonstrations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "libero_config" / "config.yaml",
        help="Machine-local config file to write.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write the config even if the LIBERO checkout is not populated.",
    )
    args = parser.parse_args()

    libero_root = args.libero_root.expanduser().resolve()
    datasets = args.datasets.expanduser().resolve()
    output = args.output.expanduser().resolve()
    paths = {
        "assets": libero_root / "assets",
        "bddl_files": libero_root / "bddl_files",
        "benchmark_root": libero_root,
        "datasets": datasets,
        "init_states": libero_root / "init_files",
    }

    missing = [f"{key}: {path}" for key, path in paths.items() if key != "datasets" and not path.exists()]
    if missing and not args.allow_missing:
        parser.error(
            "LIBERO checkout is incomplete; initialize the submodule or pass --libero-root. Missing:\n  "
            + "\n  ".join(missing)
        )

    datasets.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{key}: {json.dumps(str(path))}\n" for key, path in paths.items())
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
