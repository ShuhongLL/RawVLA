#!/usr/bin/env python
"""Render RAWVLA-Bench-Light LIBERO five-lighting preview images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
STARVLA = ROOT / "starVLA"
if STARVLA.exists() and str(STARVLA) not in sys.path:
    sys.path.insert(0, str(STARVLA))
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from rawvla_bench.libero import apply_entry_lighting, ordered_base_entries  # noqa: E402
from rawvla_bench.lighting import LIGHTING_DOMAINS, rawvla_light_xml_from_env  # noqa: E402
from rawvla_bench.manifest import (  # noqa: E402
    build_libero_manifest,
    entries_for_task,
    load_manifest,
    write_manifest,
)
from rawvla_bench.raw import make_rawvla_observation, unprocess_metadata  # noqa: E402

LIBERO_ENV_RESOLUTION = 256


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _label(image: Image.Image, text: str) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    font = _font(13)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, bbox[2] + 10, bbox[3] + 8), fill=(255, 255, 255))
    draw.text((5, 4), text, fill=(0, 0, 0), font=font)
    return out


def _sheet(items: list[tuple[str, Image.Image]], columns: int = 5) -> Image.Image:
    width, height = items[0][1].size
    rows = (len(items) + columns - 1) // columns
    pad = 8
    sheet = Image.new("RGB", (columns * width + (columns + 1) * pad, rows * height + (rows + 1) * pad), "white")
    for idx, (label, image) in enumerate(items):
        x = pad + (idx % columns) * (width + pad)
        y = pad + (idx // columns) * (height + pad)
        sheet.paste(_label(image, label), (x, y))
    return sheet


def _stats(image: np.ndarray) -> dict[str, float]:
    arr = image.astype(np.float32) / 255.0
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return {
        "mean_luma": float(luma.mean()),
        "median_luma": float(np.median(luma)),
        "dark_ratio": float((luma < 0.03).mean()),
        "clip_ratio": float((arr >= 250.0 / 255.0).any(axis=-1).mean()),
    }


def _normalize_raw_for_display(
    image: np.ndarray,
    *,
    target_p95_luma: float = 0.65,
    percentile: float = 95.0,
) -> tuple[np.ndarray, float]:
    """Exposure-normalize a RAW view for display only."""
    arr = image.astype(np.float32) / 255.0
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    reference = max(float(np.percentile(luma, percentile)), 1.0 / 255.0)
    scale = float(target_p95_luma / reference)
    normalized = np.clip(arr * np.float32(scale), 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8), scale


def _resize_float_image_to_uint8(
    image: np.ndarray,
    *,
    height: int = 224,
    width: int = 224,
    resample: int = Image.BILINEAR,
) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.shape[:2] != (height, width):
        channels = [
            np.asarray(Image.fromarray(arr[..., index], mode="F").resize((width, height), resample=resample))
            for index in range(arr.shape[-1])
        ]
        arr = np.stack(channels, axis=-1)
    return np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _render_entry(suite: str, task_id: int, entry: dict, out_dir: Path) -> dict[str, object]:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    try:
        env.seed(int(entry["base_seed"]))
        env.reset()
        rawvla_xml = rawvla_light_xml_from_env(env)
        env.reset_from_xml_string(rawvla_xml)
        env.sim.reset()
        if hasattr(env, "_rawvla_bench_lighting_baseline"):
            delattr(env, "_rawvla_bench_lighting_baseline")
        apply_entry_lighting(env, entry)
        obs = env.set_init_state(initial_states[int(entry["init_state_index"])])
    finally:
        env.close()

    domain = entry["lighting_domain"]
    row: dict[str, object] = {
        "suite": suite,
        "task_id": task_id,
        "task_description": task.language,
        "base_seed_id": entry["base_seed_id"],
        "domain": domain,
        "lighting_ev": entry["lighting_ev"],
        "lighting_scale": entry["lighting_scale"],
        "lighting_rig": entry.get("lighting_rig", "default"),
        "agentview_table_flood": entry.get("agentview_table_flood", 0.0),
        "sensor_saturation_ev": entry.get("sensor_saturation_ev", 0.0),
        "noise_seed": entry["noise_seed"],
    }

    single_root = out_dir / "single_images"
    for view_name, obs_key, view_index in (
        ("desktop_camera", "agentview_image", 0),
        ("wrist_camera", "robot0_eye_in_hand_image", 1),
    ):
        view_dir = single_root / view_name / domain
        view_dir.mkdir(parents=True, exist_ok=True)
        rgb = np.ascontiguousarray(obs[obs_key][::-1, ::-1])
        rgb224 = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR))
        # Correct deployment order: RAW/noise is formed at sensor/render
        # resolution, then the downstream policy input is resized.
        raw_native = make_rawvla_observation(
            rgb,
            noise_seed=int(entry["noise_seed"]),
            frame_index=0,
            view_index=view_index,
            representation="raw",
            sensor_saturation_ev=float(entry.get("sensor_saturation_ev", 0.0)),
        )
        default_isp_native = make_rawvla_observation(
            rgb,
            noise_seed=int(entry["noise_seed"]),
            frame_index=0,
            view_index=view_index,
            representation="default_isp",
            sensor_saturation_ev=float(entry.get("sensor_saturation_ev", 0.0)),
        )
        raw_view = _resize_float_image_to_uint8(raw_native)
        default_isp = _resize_float_image_to_uint8(default_isp_native)
        raw_normalized, raw_normalized_scale = _normalize_raw_for_display(raw_view)

        Image.fromarray(rgb224).save(view_dir / "render_rgb.png")
        Image.fromarray(raw_view).save(view_dir / "raw.png")
        Image.fromarray(raw_normalized).save(view_dir / "raw_normalized.png")
        Image.fromarray(default_isp).save(view_dir / "default_isp.png")
        row[f"{view_name}_raw_normalized_display_scale"] = raw_normalized_scale
        for prefix, image in (
            ("render_rgb", rgb224),
            ("raw", raw_view),
            ("raw_normalized", raw_normalized),
            ("default_isp", default_isp),
        ):
            for key, value in _stats(image).items():
                row[f"{view_name}_{prefix}_{key}"] = value

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "benchmark/rawvla-bench/libero_manifest_50_init_states_10000_rollouts.json",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmark/rawvla-bench/libero_lighting_preview")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--base-seed-id", type=int, default=0)
    parser.add_argument("--benchmark-seed", type=int, default=20260813)
    args = parser.parse_args()

    if not args.manifest.exists():
        write_manifest(build_libero_manifest(benchmark_seed=args.benchmark_seed), args.manifest)

    manifest = load_manifest(args.manifest)
    entries = ordered_base_entries(entries_for_task(manifest, args.suite, args.task_id), args.base_seed_id)
    if [entry["lighting_domain"] for entry in entries] != list(LIGHTING_DOMAINS):
        raise RuntimeError(f"Expected all five lighting domains, got {[entry['lighting_domain'] for entry in entries]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_render_entry(args.suite, args.task_id, entry, args.out_dir) for entry in entries]

    for view_name in ("desktop_camera", "wrist_camera"):
        rgb_items = []
        raw_items = []
        raw_normalized_items = []
        isp_items = []
        combined_items = []
        audit_items = []
        for entry in entries:
            domain = entry["lighting_domain"]
            ev = float(entry["lighting_ev"])
            view_dir = args.out_dir / "single_images" / view_name / domain
            rgb_image = Image.open(view_dir / "render_rgb.png")
            raw_image = Image.open(view_dir / "raw.png")
            raw_normalized_image = Image.open(view_dir / "raw_normalized.png")
            isp_image = Image.open(view_dir / "default_isp.png")
            raw_normalized_scale = next(
                float(row[f"{view_name}_raw_normalized_display_scale"])
                for row in rows
                if row["domain"] == domain
            )
            rgb_items.append((f"{domain} {ev:+.2f}EV render RGB", rgb_image))
            raw_items.append((f"{domain} {ev:+.2f}EV raw", raw_image))
            raw_normalized_items.append((f"{domain} raw norm x{raw_normalized_scale:.1f}", raw_normalized_image))
            isp_items.append((f"{domain} {ev:+.2f}EV isp", isp_image))
            combined_items.extend([(f"{domain} raw", raw_image), (f"{domain} isp", isp_image)])
            audit_items.append((f"{domain} render RGB", rgb_image))
        audit_items.extend(raw_items)
        audit_items.extend(raw_normalized_items)
        _sheet(rgb_items).save(args.out_dir / f"{view_name}_render_rgb_sheet.png")
        _sheet(raw_items).save(args.out_dir / f"{view_name}_raw_sheet.png")
        _sheet(raw_normalized_items).save(args.out_dir / f"{view_name}_raw_normalized_sheet.png")
        _sheet(isp_items).save(args.out_dir / f"{view_name}_default_isp_sheet.png")
        _sheet(combined_items, columns=5).save(args.out_dir / f"{view_name}.png")
        _sheet(audit_items, columns=5).save(args.out_dir / f"{view_name}_rgb_raw_normalized_audit.png")

    with open(args.out_dir / "stats.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "preview_manifest_entries.json").write_text(json.dumps(rows, indent=2, sort_keys=True), "utf-8")
    (args.out_dir / "unprocess_metadata.json").write_text(json.dumps(unprocess_metadata(), indent=2, sort_keys=True), "utf-8")
    print(f"Wrote preview images to {args.out_dir}")


if __name__ == "__main__":
    main()
