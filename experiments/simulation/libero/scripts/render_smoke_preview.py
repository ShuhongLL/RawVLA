#!/usr/bin/env python
"""Render a small RAWVLA-Bench-Light smoke preview across all LIBERO suites."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
STARVLA = ROOT / "starVLA"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for path in (STARVLA, PACKAGE_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "libero_config"))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from rawvla_bench.libero import apply_entry_lighting, ordered_base_entries  # noqa: E402
from rawvla_bench.lighting import LIGHTING_DOMAINS, rawvla_light_xml_from_env  # noqa: E402
from rawvla_bench.manifest import entries_for_task, load_manifest  # noqa: E402
from rawvla_bench.raw import make_rawvla_observation  # noqa: E402

LIBERO_ENV_RESOLUTION = 256
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _labeled(image: Image.Image, label: str, size: int = 12) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    font = _font(size)
    bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=2)
    draw.rectangle((0, 0, min(out.width, bbox[2] + 8), bbox[3] + 8), fill=(255, 255, 255))
    draw.multiline_text((4, 4), label, fill=(0, 0, 0), font=font, spacing=2)
    return out


def _sheet(items: list[tuple[str, Image.Image]], columns: int) -> Image.Image:
    width, height = items[0][1].size
    pad = 8
    rows = (len(items) + columns - 1) // columns
    out = Image.new("RGB", (columns * width + (columns + 1) * pad, rows * height + (rows + 1) * pad), "white")
    for idx, (label, image) in enumerate(items):
        x = pad + (idx % columns) * (width + pad)
        y = pad + (idx // columns) * (height + pad)
        out.paste(_labeled(image, label), (x, y))
    return out


def _stack_sheets(paths: list[Path], labels: list[str]) -> Image.Image:
    sheets = [Image.open(path).convert("RGB") for path in paths]
    width = max(sheet.width for sheet in sheets)
    pad = 10
    header = 28
    total_height = sum(sheet.height + header for sheet in sheets) + (len(sheets) + 1) * pad
    out = Image.new("RGB", (width + 2 * pad, total_height), "white")
    draw = ImageDraw.Draw(out)
    font = _font(16)
    y = pad
    for label, sheet in zip(labels, sheets, strict=True):
        draw.text((pad, y), label, fill=(0, 0, 0), font=font)
        y += header
        out.paste(sheet, (pad, y))
        y += sheet.height + pad
    return out


def _resize_uint8(image: np.ndarray, size: int = 224) -> np.ndarray:
    return np.asarray(Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((size, size), Image.BILINEAR))


def _stats(image: np.ndarray) -> dict[str, float]:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return {
        "mean_luma": float(luma.mean()),
        "median_luma": float(np.median(luma)),
        "dark_ratio": float((luma < 0.03).mean()),
        "clip_ratio": float((arr >= 250.0 / 255.0).any(axis=-1).mean()),
    }


def _select_episode_groups(manifest: dict, suites: tuple[str, ...], per_suite: int, seed: int) -> dict[str, list[tuple[int, int]]]:
    selected = {}
    for suite in suites:
        groups = sorted({(int(entry["task_id"]), int(entry["base_seed_id"])) for entry in manifest["entries"] if entry["suite"] == suite})
        rng = random.Random(f"{seed}:{suite}")
        selected[suite] = rng.sample(groups, per_suite)
    return selected


def _render_group(
    suite_name: str,
    task_id: int,
    base_seed_id: int,
    entries: list[dict],
    out_dir: Path,
) -> list[dict[str, object]]:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    group_dir = out_dir / suite_name / f"task_{task_id:02d}_base_{base_seed_id:02d}"
    group_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    view_items = {
        "agentview": {"raw": [], "isp": [], "rgb": [], "raw_vs_original": []},
        "wrist": {"raw": [], "isp": [], "rgb": [], "raw_vs_original": []},
    }

    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    try:
        reference_entry = entries[0]
        env.seed(int(reference_entry["base_seed"]))
        original_obs = env.reset()
        rawvla_xml = rawvla_light_xml_from_env(env)
        original_obs = env.set_init_state(initial_states[int(reference_entry["init_state_index"])])
        original_agentview = _resize_uint8(np.ascontiguousarray(original_obs["agentview_image"][::-1, ::-1]))
        original_wrist = _resize_uint8(np.ascontiguousarray(original_obs["robot0_eye_in_hand_image"][::-1, ::-1]))
        Image.fromarray(original_agentview).save(group_dir / "original_agentview_rgb.png")
        Image.fromarray(original_wrist).save(group_dir / "original_wrist_rgb.png")
        original_label = "Original RGB\nunmodified XML\nno RAW lights"
        original_items = {
            "agentview": (original_label, Image.fromarray(original_agentview)),
            "wrist": (original_label, Image.fromarray(original_wrist)),
        }

        for entry in entries:
            env.seed(int(entry["base_seed"]))
            env.reset_from_xml_string(rawvla_xml)
            env.sim.reset()
            if hasattr(env, "_rawvla_bench_lighting_baseline"):
                delattr(env, "_rawvla_bench_lighting_baseline")
            lighting_scale = apply_entry_lighting(env, entry)
            obs = env.set_init_state(initial_states[int(entry["init_state_index"])])
            row_base = {
                "suite": suite_name,
                "task_id": task_id,
                "base_seed_id": base_seed_id,
                "task_description": task.language,
                "domain": entry["lighting_domain"],
                "lighting_ev": float(entry["lighting_ev"]),
                "lighting_scale": float(lighting_scale),
                "lighting_rig": entry.get("lighting_rig", "default"),
                "agentview_table_flood": float(entry.get("agentview_table_flood", 0.0)),
                "sensor_saturation_ev": float(entry.get("sensor_saturation_ev", 0.0)),
                "noise_seed": int(entry["noise_seed"]),
            }
            for view_name, obs_key, view_index in (
                ("agentview", "agentview_image", 0),
                ("wrist", "robot0_eye_in_hand_image", 1),
            ):
                rgb = np.ascontiguousarray(obs[obs_key][::-1, ::-1])
                rgb224 = _resize_uint8(rgb)
                raw = _resize_uint8(
                    make_rawvla_observation(
                        rgb,
                        noise_seed=int(entry["noise_seed"]),
                        frame_index=0,
                        view_index=view_index,
                        representation="raw",
                        sensor_saturation_ev=float(entry.get("sensor_saturation_ev", 0.0)),
                        exposure_ev=0.0,
                    )
                )
                isp = _resize_uint8(
                    make_rawvla_observation(
                        rgb,
                        noise_seed=int(entry["noise_seed"]),
                        frame_index=0,
                        view_index=view_index,
                        representation="default_isp",
                        sensor_saturation_ev=float(entry.get("sensor_saturation_ev", 0.0)),
                        exposure_ev=0.0,
                    )
                )
                prefix = f"{entry['lighting_domain']}_{view_name}"
                Image.fromarray(rgb224).save(group_dir / f"{prefix}_render_rgb.png")
                Image.fromarray(raw).save(group_dir / f"{prefix}_raw.png")
                Image.fromarray(isp).save(group_dir / f"{prefix}_default_isp_ev0.png")
                label = (
                    f"{entry['lighting_domain']}\n"
                    f"EV {float(entry['lighting_ev']):+.2f}\n"
                    f"flood {float(entry.get('agentview_table_flood', 0.0)):.2f}\n"
                    f"sat {float(entry.get('sensor_saturation_ev', 0.0)):.2f}"
                )
                view_items[view_name]["raw"].append((label, Image.fromarray(raw)))
                view_items[view_name]["isp"].append((label, Image.fromarray(isp)))
                view_items[view_name]["rgb"].append((label, Image.fromarray(rgb224)))
                view_items[view_name]["raw_vs_original"].append((label, Image.fromarray(raw)))
                stats_row = dict(row_base)
                stats_row["view"] = view_name
                for stat_name, value in _stats(raw).items():
                    stats_row[f"raw_{stat_name}"] = value
                for stat_name, value in _stats(isp).items():
                    stats_row[f"default_isp_ev0_{stat_name}"] = value
                rows.append(stats_row)
    finally:
        env.close()

    for view_name in ("agentview", "wrist"):
        raw_items = view_items[view_name]["raw"]
        isp_items = view_items[view_name]["isp"]
        rgb_items = view_items[view_name]["rgb"]
        raw_vs_original_items = view_items[view_name]["raw_vs_original"]
        original_item = original_items[view_name]
        _sheet([original_item], columns=1).save(group_dir / f"{view_name}_original_rgb.png")
        _sheet(raw_items, columns=5).save(group_dir / f"{view_name}_raw_5domains.png")
        _sheet(isp_items, columns=5).save(group_dir / f"{view_name}_default_isp_ev0_5domains.png")
        _sheet(rgb_items, columns=5).save(group_dir / f"{view_name}_render_rgb_5domains.png")
        _sheet([original_item] + raw_vs_original_items, columns=6).save(
            group_dir / f"{view_name}_original_rgb_vs_raw_5domains.png"
        )
        _sheet(raw_items + isp_items, columns=5).save(group_dir / f"{view_name}_raw_and_default_isp_ev0.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/simulation/libero/manifests/libero_manifest_50_init_states_10000_rollouts.json",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/simulation/libero/smoke_preview")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--per-suite", type=int, default=2)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    selections = _select_episode_groups(manifest, SUITES, args.per_suite, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "selection.json").write_text(json.dumps(selections, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    for suite_name in SUITES:
        for task_id, base_seed_id in selections[suite_name]:
            suite_entries = entries_for_task(manifest, suite_name, task_id)
            group_entries = ordered_base_entries(suite_entries, base_seed_id)
            if [entry["lighting_domain"] for entry in group_entries] != list(LIGHTING_DOMAINS):
                raise RuntimeError(f"Missing lighting domains for {suite_name} task={task_id} base={base_seed_id}")
            rows.extend(_render_group(suite_name, task_id, base_seed_id, group_entries, args.out_dir))

    with (args.out_dir / "stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "stats.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    group_dirs = []
    labels = []
    for suite_name in SUITES:
        for task_id, base_seed_id in selections[suite_name]:
            group_dirs.append(args.out_dir / suite_name / f"task_{task_id:02d}_base_{base_seed_id:02d}")
            labels.append(f"{suite_name}/task_{task_id:02d}_base_{base_seed_id:02d}")
    for view_name in ("agentview", "wrist"):
        for filename in (
            f"{view_name}_original_rgb_vs_raw_5domains.png",
            f"{view_name}_original_rgb.png",
            f"{view_name}_raw_5domains.png",
            f"{view_name}_default_isp_ev0_5domains.png",
            f"{view_name}_render_rgb_5domains.png",
            f"{view_name}_raw_and_default_isp_ev0.png",
        ):
            _stack_sheets([group_dir / filename for group_dir in group_dirs], labels).save(args.out_dir / f"all_{filename}")
    print(json.dumps({"out_dir": str(args.out_dir), "selections": selections}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
