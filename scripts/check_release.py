#!/usr/bin/env python3
"""Dependency-light checks for a RAW-VLA public source release."""

from __future__ import annotations

import argparse
import compileall
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "baselines/rawvla/rawvla.py",
    "baselines/rawvla/smoke_test.py",
    "benchmark/rawvla-bench/rawvla_bench/raw.py",
    "benchmark/rawvla-bench/scripts/run_libero_models.sh",
    "experiments/isp_perturbation/README.md",
    "experiments/real_robot/README.md",
    "starVLA/starVLA/model/modules/raw_frontend/registry.py",
    "scripts/replay_libero_train_rawvla_light_npz.py",
    "scripts/replay_robotwin2_clean_training.py",
    "scripts/replay_robotwin2_paired_lighting_training.py",
    "scripts/configure_libero.py",
    "configs/rawvla_qwen3_oft_libero.yaml",
    "libero_config/config.yaml.example",
)
RUNTIME_SUFFIXES = {".py", ".sh"}
SKIP_PARTS = {".git", "__pycache__"}
SUBMODULE_ROOTS = (
    ROOT / "starVLA",
    ROOT / "LIBERO-git",
    ROOT / "third_party" / "openvla-oft",
    ROOT / "third_party" / "Isaac-GR00T",
    ROOT / "third_party" / "RoboTwin",
    ROOT / "third_party" / "FastWAM",
)
PRIVATE_PATTERNS = (
    re.compile("/" + "DRoboticsAILab/"),
    re.compile(r"/home/"),
    re.compile(r"/mnt/"),
    re.compile(r"/RAW-VLA/"),
    re.compile("/" + "llm-serving-pvc/"),
    re.compile("/tmp/" + "scratch-space/"),
)
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
)


def run(command: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True)


def iter_runtime_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in RUNTIME_SUFFIXES:
            continue
        if path == Path(__file__).resolve():
            continue
        if any(path.is_relative_to(submodule_root) for submodule_root in SUBMODULE_ROOTS):
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="also require initialized submodules")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    if "git@" in gitmodules:
        errors.append(".gitmodules contains an SSH-only URL")

    for path in iter_runtime_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                errors.append(f"machine-private path in {path.relative_to(ROOT)}")
                break

    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path == Path(__file__).resolve():
            continue
        if any(path.is_relative_to(submodule_root) for submodule_root in SUBMODULE_ROOTS):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        duplicated_placeholder = "/path/to/" + "path/to/"
        if duplicated_placeholder in text:
            errors.append(f"duplicated placeholder path in {path.relative_to(ROOT)}")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"possible secret in {path.relative_to(ROOT)}")

    python_roots = [
        ROOT / "baselines",
        ROOT / "benchmark" / "rawvla-bench",
        ROOT / "experiments",
        ROOT / "scripts",
        ROOT / "finetune",
        ROOT / "tools",
    ]
    for python_root in python_roots:
        if python_root.exists() and not compileall.compile_dir(python_root, quiet=1, force=True):
            errors.append(f"Python syntax check failed under {python_root.relative_to(ROOT)}")

    for path in iter_runtime_files():
        if path.suffix != ".sh":
            continue
        checked = run(["bash", "-n", str(path)])
        if checked.returncode:
            errors.append(f"shell syntax check failed: {path.relative_to(ROOT)}: {checked.stderr.strip()}")

    submodules = run(["git", "config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"])
    if submodules.returncode == 0:
        for line in submodules.stdout.splitlines():
            relative = line.split(maxsplit=1)[1]
            marker = ROOT / relative
            initialized = (marker / ".git").exists() or bool(list(marker.glob("*")))
            if not initialized:
                message = f"submodule not initialized: {relative}"
                (errors if args.strict else warnings).append(message)

    if not (ROOT / "LICENSE").exists():
        warnings.append("no root LICENSE selected yet")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print(f"release check failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
