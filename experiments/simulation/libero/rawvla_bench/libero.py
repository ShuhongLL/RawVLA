"""LIBERO-specific helpers for RAWVLA-Bench."""

from __future__ import annotations

from typing import Any

from .lighting import LIGHTING_DOMAINS, apply_lighting_ev, prepare_lighting_rig
from .manifest import entries_for_task, load_manifest


def load_episode_plan(manifest_path: str, suite: str, task_id: int) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    entries = entries_for_task(manifest, suite, task_id)
    if not entries:
        raise ValueError(f"No RAWVLA-Bench entries for suite={suite!r}, task_id={task_id}")
    return entries


def ordered_base_entries(entries: list[dict[str, Any]], base_seed_id: int) -> list[dict[str, Any]]:
    selected = [entry for entry in entries if int(entry["base_seed_id"]) == int(base_seed_id)]
    domain_order = {domain: idx for idx, domain in enumerate(LIGHTING_DOMAINS)}
    return sorted(selected, key=lambda entry: domain_order[entry["lighting_domain"]])


def apply_entry_lighting(env: object, entry: dict[str, Any]) -> float:
    return apply_lighting_ev(
        env,
        float(entry["lighting_ev"]),
        str(entry.get("lighting_rig", "default")),
        float(entry.get("agentview_table_flood", 0.0)),
    )


def prepare_entry_lighting(env: object, entry: dict[str, Any]) -> None:
    prepare_lighting_rig(env, str(entry.get("lighting_rig", "default")))
