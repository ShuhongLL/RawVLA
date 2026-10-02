#!/usr/bin/env python3
"""Check one RAW-VLA runtime environment with its active Python interpreter."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES: dict[str, tuple[str, ...]] = {
    "rawvla": ("torch", "numpy"),
    "starvla-libero": (
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "numpy",
        "h5py",
        "mujoco",
        "robosuite",
        "bddl",
        "libero",
        "omegaconf",
        "starVLA",
        "rawvla_bench",
    ),
    "robotwin": (
        "torch",
        "torchvision",
        "sapien",
        "mplib",
        "transforms3d",
        "scipy",
        "skimage",
        "gymnasium",
        "trimesh",
        "open3d",
        "curobo",
    ),
    "openvla-oft": ("torch", "torchvision", "transformers", "tensorflow", "tensorflow_datasets", "prismatic"),
    "fastwam": ("torch", "torchvision", "transformers", "accelerate", "fastwam"),
    "groot": ("torch", "torchvision", "transformers", "gr00t"),
}


def add_source_paths() -> None:
    paths = (
        ROOT,
        ROOT / "starVLA",
        ROOT / "LIBERO-git",
        ROOT / "third_party" / "openvla-oft",
        ROOT / "third_party" / "RoboTwin",
        ROOT / "third_party" / "FastWAM" / "src",
        ROOT / "third_party" / "Isaac-GR00T",
        ROOT / "benchmark" / "rawvla-bench",
    )
    for path in reversed(paths):
        if path.exists():
            sys.path.insert(0, str(path))


def check_imports(profile: str) -> list[str]:
    failures: list[str] = []
    print(f"python {sys.version.split()[0]} ({sys.executable})")
    for name in PROFILES[profile]:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # Import failures are the point of this diagnostic.
            message = f"{name}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(f"FAIL {message}")
        else:
            print(f"OK   {name} {getattr(module, '__version__', '')}".rstrip())
    try:
        import torch

        print(
            f"torch CUDA build={torch.version.cuda}, available={torch.cuda.is_available()}, "
            f"devices={torch.cuda.device_count()}"
        )
    except Exception:
        pass
    return failures


def rawvla_smoke() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    return subprocess.run([sys.executable, "-m", "baselines.rawvla.smoke_test"], cwd=ROOT, env=env).returncode


def libero_render_smoke() -> None:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
    try:
        observation = env.reset()
        if not isinstance(observation, dict) or not observation:
            raise RuntimeError("LIBERO reset returned no observations")
        print(f"LIBERO render/reset passed ({len(observation)} observation keys)")
    finally:
        env.close()


def robotwin_render_smoke() -> None:
    import sapien

    renderer = sapien.SapienRenderer()
    del renderer
    print("SAPIEN renderer construction passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=sorted(PROFILES))
    parser.add_argument("--render", action="store_true", help="Also create a LIBERO or SAPIEN renderer where supported.")
    args = parser.parse_args()
    add_source_paths()
    failures = check_imports(args.profile)
    if failures:
        print(f"environment check failed: {len(failures)} import(s)", file=sys.stderr)
        return 1
    if args.profile == "rawvla" and rawvla_smoke():
        return 1
    if args.render and args.profile == "starvla-libero":
        libero_render_smoke()
    elif args.render and args.profile == "robotwin":
        robotwin_render_smoke()
    elif args.render:
        parser.error(f"--render is not supported for {args.profile}")
    print(f"{args.profile} environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
