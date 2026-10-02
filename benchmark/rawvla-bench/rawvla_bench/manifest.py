"""Frozen manifest construction for RAWVLA-Bench-Light LIBERO v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .lighting import (
    DEFAULT_EV_RANGES,
    LIGHTING_DOMAINS,
    LIGHTING_EV_DISTRIBUTION,
    lighting_domain_configs_for_manifest,
    lighting_parameters_for_sample,
)

BENCHMARK_NAME = "rawvla-bench-light-libero-v1"

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def stable_uint32(*parts: object) -> int:
    text = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _sample_unit(
    benchmark_seed: int,
    suite: str,
    task_id: int,
    base_seed: int,
    domain: str,
) -> float:
    rng = np.random.default_rng(stable_uint32("lighting_ev", benchmark_seed, suite, task_id, base_seed, domain))
    return float(rng.random())


def _sample_ev(
    sample_unit: float,
    domain: str,
    ev_ranges: dict[str, tuple[float, float]],
) -> float:
    low, high = ev_ranges[domain]
    return float(low + float(sample_unit) * (high - low))


def build_libero_manifest(
    *,
    benchmark_seed: int = 20260813,
    suites: tuple[str, ...] = LIBERO_SUITES,
    tasks_per_suite: int = 10,
    base_seeds_per_task: int = 10,
    ev_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Build the paired 10 base-seed x 5 lighting-domain LIBERO manifest."""
    ranges = dict(DEFAULT_EV_RANGES if ev_ranges is None else ev_ranges)
    entries: list[dict[str, Any]] = []
    for suite in suites:
        for task_id in range(tasks_per_suite):
            for base_seed_id in range(base_seeds_per_task):
                base_seed = int(base_seed_id)
                for domain_idx, domain in enumerate(LIGHTING_DOMAINS):
                    episode_idx = base_seed_id * len(LIGHTING_DOMAINS) + domain_idx
                    sample_unit = _sample_unit(benchmark_seed, suite, task_id, base_seed, domain)
                    lighting_ev = _sample_ev(sample_unit, domain, ranges)
                    lighting_params = lighting_parameters_for_sample(domain, lighting_ev, sample_unit=sample_unit)
                    entries.append(
                        {
                            "benchmark": BENCHMARK_NAME,
                            "suite": suite,
                            "task_id": task_id,
                            "episode_idx": episode_idx,
                            "base_seed_id": base_seed_id,
                            "base_seed": base_seed,
                            "init_state_index": base_seed_id,
                            "lighting_domain": domain,
                            "lighting_sample_unit": sample_unit,
                            **lighting_params,
                            "noise_seed": stable_uint32("raw_noise", benchmark_seed, suite, task_id, base_seed, domain),
                        }
                    )
    return {
        "benchmark": BENCHMARK_NAME,
        "version": 1,
        "benchmark_seed": benchmark_seed,
        "suites": list(suites),
        "tasks_per_suite": tasks_per_suite,
        "base_seeds_per_task": base_seeds_per_task,
        "lighting_domains": list(LIGHTING_DOMAINS),
        "lighting_distribution": LIGHTING_EV_DISTRIBUTION,
        "ev_ranges": {key: list(value) for key, value in ranges.items()},
        "lighting_domain_configs": lighting_domain_configs_for_manifest(ranges),
        "entries": entries,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("benchmark") != BENCHMARK_NAME:
        raise ValueError(f"Expected benchmark={BENCHMARK_NAME!r}, got {manifest.get('benchmark')!r}")
    return manifest


def entries_for_task(manifest: dict[str, Any], suite: str, task_id: int) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in manifest["entries"]
        if entry["suite"] == suite and int(entry["task_id"]) == int(task_id)
    ]
    return sorted(entries, key=lambda entry: int(entry["episode_idx"]))
