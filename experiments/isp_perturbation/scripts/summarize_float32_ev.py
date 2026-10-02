#!/usr/bin/env python3
"""Summarize the legacy synthetic float32 EV perturbation matrix."""
import argparse
import json
import os
from pathlib import Path
import time


def render(root: Path) -> str:
    rows = []
    for ledger in sorted(root.glob("*/*/ev_*/*/episodes.jsonl")):
        records = {}
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                records[(rec["task_id"], rec["episode_idx"])] = bool(rec["success"])
            except (KeyError, json.JSONDecodeError):
                pass
        rep_dir = ledger.parent
        ev_dir, suite_dir, model_dir = rep_dir.parent, rep_dir.parent.parent, rep_dir.parent.parent.parent
        total = len(records)
        success = sum(records.values())
        audit = rep_dir / "image_audit.jsonl"
        preset_zero = (rep_dir / "preset_zero").exists()
        audit_ok = False
        if audit.exists():
            stages = {json.loads(x)["stage"] for x in audit.read_text().splitlines() if x.strip()}
            audit_ok = (
                {"policy_server_received", "qwen_processor_input"} <= stages
                or {"policy_server_received", "pi_model_input"} <= stages
                or {"policy_server_received", "world_model_input"} <= stages
            )
        rows.append(
            (model_dir.name, suite_dir.name, ev_dir.name.removeprefix("ev_"), rep_dir.name,
             success, total, 100.0 * success / total if total else 0.0,
             "preset" if preset_zero else ("yes" if audit_ok else "no"),
             (rep_dir / "job.done").exists())
        )
    lines = [
        "# LIBERO float32 EV zero-shot results",
        "",
        f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "| Model | Suite | EV | Input | Success | Episodes | SR | Float audit | Done |",
        "|---|---|---:|---|---:|---:|---:|:---:|:---:|",
    ]
    lines += [
        f"| {m} | {s} | {ev} | {r} | {ok} | {n} | {sr:.2f}% | {audit} | {'yes' if done else 'no'} |"
        for m, s, ev, r, ok, n, sr, audit, done in rows
    ]
    return "\n".join(lines) + "\n"


def write(root: Path) -> None:
    target = root / "RESULTS.md"
    temp = target.with_suffix(".tmp")
    temp.write_text(render(root), encoding="utf-8")
    os.replace(temp, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        write(args.root)
        if not args.watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
