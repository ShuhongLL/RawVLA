#!/usr/bin/env python3
"""Download StarVLA LIBERO finetuned checkpoints with resumable aria2.

Run with:
  conda activate starVLA
  python tools/checkpoints/download_starvla_libero.py

By default the script downloads only StarVLA/Qwen3-VL-PI-LIBERO-4in1 plus the
base VLM referenced by its config.yaml. Pass --all to fetch every listed LIBERO
finetune repo and all known base VLM repos.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata, list_repo_files


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("STARVLA_ROOT", REPO_ROOT / "starVLA")).expanduser().resolve()
PRETRAINED = ROOT / "playground" / "Pretrained_models"

FINETUNED = [
    ("StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1", "checkpoints/steps_30000_pytorch_model.pt"),
    ("StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1", "checkpoints/steps_30000_pytorch_model.pt"),
    ("StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1", "checkpoints/steps_30000_pytorch_model.pt"),
    ("StarVLA/Qwen3-VL-PI-LIBERO-4in1", "checkpoints/steps_100000_pytorch_model.pt"),
    ("StarVLA/Qwen3-VL-OFT-LIBERO-4in1", "checkpoints/steps_50000_pytorch_model.pt"),
]

BASE_MODELS = [
    ("Qwen/Qwen2.5-VL-3B-Instruct", PRETRAINED / "Qwen2.5-VL-3B-Instruct"),
    ("StarVLA/Qwen2.5-VL-3B-Instruct-Action", PRETRAINED / "Qwen2.5-VL-3B-Instruct-Action"),
    ("Qwen/Qwen3-VL-4B-Instruct", PRETRAINED / "Qwen3-VL-4B-Instruct"),
]

BASE_BY_FINETUNE = {
    "StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1": ["Qwen/Qwen2.5-VL-3B-Instruct"],
    "StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1": ["StarVLA/Qwen2.5-VL-3B-Instruct-Action"],
    "StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1": ["Qwen/Qwen2.5-VL-3B-Instruct"],
    "StarVLA/Qwen3-VL-PI-LIBERO-4in1": ["Qwen/Qwen3-VL-4B-Instruct"],
    "StarVLA/Qwen3-VL-OFT-LIBERO-4in1": ["Qwen/Qwen3-VL-4B-Instruct"],
}


def hf_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    return env


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def download_small_repo_files(repo_id: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_repo_files(repo_id) if not f.startswith("checkpoints/")]
    for filename in files:
        target = dest / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"Downloading small file {repo_id}:{filename}", flush=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            force_download=False,
            resume_download=True,
        )


def download_large_file(repo_id: str, filename: str, dest: Path) -> None:
    target = dest / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo_id, filename)
    meta = get_hf_file_metadata(url, timeout=30)
    if target.exists() and target.stat().st_size == meta.size:
        print(f"Already complete: {target}", flush=True)
        return
    run(
        [
            "aria2c",
            "-c",
            "-x",
            "16",
            "-s",
            "16",
            "-k",
            "1M",
            "--summary-interval=30",
            "--max-tries=0",
            "--retry-wait=5",
            "--timeout=60",
            "--connect-timeout=30",
            "-d",
            str(target.parent),
            "-o",
            target.name,
            meta.location,
        ],
        env=os.environ.copy(),
    )


def download_model_repo(repo_id: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    run(
        [
            "huggingface-cli",
            "download",
            repo_id,
            "--local-dir",
            str(dest),
            "--max-workers",
            "4",
        ],
        env=hf_env(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="StarVLA/Qwen3-VL-PI-LIBERO-4in1",
        help="Finetuned StarVLA repo to download. Ignored when --all is set.",
    )
    parser.add_argument("--all", action="store_true", help="Download all known LIBERO finetuned repos.")
    parser.add_argument("--skip-base", action="store_true", help="Do not download base VLM repos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finetuned_by_repo = dict(FINETUNED)
    if args.all:
        selected = FINETUNED
    else:
        if args.repo_id not in finetuned_by_repo:
            known = ", ".join(sorted(finetuned_by_repo))
            raise ValueError(f"Unknown repo-id {args.repo_id!r}. Known repos: {known}")
        selected = [(args.repo_id, finetuned_by_repo[args.repo_id])]

    for repo_id, checkpoint in selected:
        dest = PRETRAINED / "StarVLA" / repo_id.split("/", 1)[1]
        print(f"\n=== {repo_id} ===", flush=True)
        download_small_repo_files(repo_id, dest)
        download_large_file(repo_id, checkpoint, dest)

    if args.skip_base:
        return

    base_by_repo = dict(BASE_MODELS)
    if args.all:
        base_repo_ids = [repo_id for repo_id, _ in BASE_MODELS]
    else:
        base_repo_ids = BASE_BY_FINETUNE.get(args.repo_id, [])

    for repo_id in base_repo_ids:
        dest = base_by_repo[repo_id]
        print(f"\n=== base {repo_id} ===", flush=True)
        download_model_repo(repo_id, dest)


if __name__ == "__main__":
    main()
