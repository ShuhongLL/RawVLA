#!/usr/bin/env python3
import json
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def main(root: pathlib.Path) -> None:
    rows = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "control"):
        for suite_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            successes = total = 0
            ledger = suite_dir / "episodes.jsonl"
            if ledger.exists():
                episodes = {}
                for line in ledger.read_text(encoding="utf-8").splitlines():
                    try:
                        item = json.loads(line)
                        episodes[(item["task_id"], item["episode_idx"])] = bool(item["success"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
                total = len(episodes)
                successes = sum(episodes.values())
            else:
                for result in suite_dir.glob("task_*.json"):
                    try:
                        item = json.loads(result.read_text(encoding="utf-8"))
                        successes += int(item["total_successes"])
                        total += int(item["total_episodes"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        pass
            rows.append((model_dir.name, suite_dir.name, successes, total))

    print(f"{'MODEL':48} {'SUITE':16} {'SUCCESS':>8} {'TOTAL':>8} {'SR':>9}")
    grand_s = grand_n = 0
    by_model = {}
    for model, suite, successes, total in rows:
        rate = 100.0 * successes / total if total else 0.0
        print(f"{model:48} {suite:16} {successes:8d} {total:8d} {rate:8.2f}%")
        grand_s += successes
        grand_n += total
        acc = by_model.setdefault(model, [0, 0])
        acc[0] += successes
        acc[1] += total
    print("\nPER MODEL")
    for model, (successes, total) in sorted(by_model.items()):
        rate = 100.0 * successes / total if total else 0.0
        print(f"{model:48} {successes:8d}/{total:<8d} {rate:8.2f}%")
    rate = 100.0 * grand_s / grand_n if grand_n else 0.0
    print(f"\nALL {grand_s}/{grand_n} {rate:.2f}%")


if __name__ == "__main__":
    default_root = REPO_ROOT / "results" / "rawvla-bench-light-libero"
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default_root)
