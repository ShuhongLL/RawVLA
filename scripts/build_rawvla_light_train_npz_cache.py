#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAWVLA_BENCH_ROOT = ROOT / "experiments" / "simulation" / "libero"
if str(RAWVLA_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RAWVLA_BENCH_ROOT))

from rawvla_bench.lighting import DEFAULT_EV_RANGES, LIGHTING_DOMAINS, lighting_parameters_for_sample  # noqa: E402
from rawvla_bench.raw import make_rawvla_observation, stable_uint32, unprocess_metadata  # noqa: E402


RLDS_TO_LEROBOT_NAME = {
    "libero_10_no_noops": "libero_10_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops": "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object_no_noops": "libero_object_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops": "libero_spatial_no_noops_1.0.0_lerobot",
}


@dataclass(frozen=True)
class ShardTask:
    rlds_dataset_name: str
    lerobot_dataset_name: str
    shard_path: str
    shard_index: int
    episode_offset: int
    episode_count: int


def _decode_jpeg(raw: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if pos >= len(data):
            raise ValueError("Truncated protobuf varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("Protobuf varint is too long")


def _iter_proto_fields(data: bytes) -> Iterator[tuple[int, int, bytes]]:
    pos = 0
    size = len(data)
    while pos < size:
        tag, pos = _read_varint(data, pos)
        field_no = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            start = pos
            _, pos = _read_varint(data, pos)
            yield field_no, wire_type, data[start:pos]
        elif wire_type == 1:
            end = pos + 8
            yield field_no, wire_type, data[pos:end]
            pos = end
        elif wire_type == 2:
            length, pos = _read_varint(data, pos)
            end = pos + length
            yield field_no, wire_type, data[pos:end]
            pos = end
        elif wire_type == 5:
            end = pos + 4
            yield field_no, wire_type, data[pos:end]
            pos = end
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")


def _bytes_list_values(bytes_list_message: bytes) -> list[bytes]:
    return [raw for field_no, wire_type, raw in _iter_proto_fields(bytes_list_message) if field_no == 1 and wire_type == 2]


def _feature_bytes_values(feature_message: bytes) -> list[bytes]:
    for field_no, wire_type, raw in _iter_proto_fields(feature_message):
        if field_no == 1 and wire_type == 2:
            return _bytes_list_values(raw)
    return []


def _feature_list_values(feature_list_message: bytes) -> list[bytes]:
    values: list[bytes] = []
    for field_no, wire_type, raw in _iter_proto_fields(feature_list_message):
        if field_no == 1 and wire_type == 2:
            feature_values = _feature_bytes_values(raw)
            if feature_values:
                values.append(feature_values[0])
    return values


def _parse_feature_list_entry(entry_message: bytes) -> tuple[str | None, list[bytes] | None]:
    key = None
    values = None
    for field_no, wire_type, raw in _iter_proto_fields(entry_message):
        if field_no == 1 and wire_type == 2:
            key = raw.decode("utf-8")
        elif field_no == 2 and wire_type == 2:
            values = _feature_list_values(raw)
    return key, values


def _extract_sequence_jpegs(serialized: bytes) -> dict[str, list[bytes]]:
    wanted = {"steps/observation/image", "steps/observation/wrist_image"}
    result: dict[str, list[bytes]] = {}
    for field_no, wire_type, raw in _iter_proto_fields(serialized):
        if field_no != 2 or wire_type != 2:
            continue
        for entry_field_no, entry_wire_type, entry_raw in _iter_proto_fields(raw):
            if entry_field_no != 1 or entry_wire_type != 2:
                continue
            key, values = _parse_feature_list_entry(entry_raw)
            if key in wanted and values is not None:
                result[key] = values
    missing = wanted.difference(result)
    if missing:
        raise ValueError(f"SequenceExample missing feature_list keys: {sorted(missing)}")
    return result


def _parse_feature_entry(entry_message: bytes) -> tuple[str | None, list[bytes] | None]:
    key = None
    values = None
    for field_no, wire_type, raw in _iter_proto_fields(entry_message):
        if field_no == 1 and wire_type == 2:
            key = raw.decode("utf-8")
        elif field_no == 2 and wire_type == 2:
            values = _feature_bytes_values(raw)
    return key, values


def _extract_example_jpegs(serialized: bytes) -> dict[str, list[bytes]]:
    wanted = {"steps/observation/image", "steps/observation/wrist_image"}
    result: dict[str, list[bytes]] = {}
    for field_no, wire_type, raw in _iter_proto_fields(serialized):
        if field_no != 1 or wire_type != 2:
            continue
        for entry_field_no, entry_wire_type, entry_raw in _iter_proto_fields(raw):
            if entry_field_no != 1 or entry_wire_type != 2:
                continue
            key, values = _parse_feature_entry(entry_raw)
            if key in wanted and values is not None:
                result[key] = values
    missing = wanted.difference(result)
    if missing:
        raise ValueError(f"Example missing feature keys: {sorted(missing)}")
    return result


def _extract_episode_jpegs(serialized: bytes) -> dict[str, list[bytes]]:
    try:
        return _extract_sequence_jpegs(serialized)
    except ValueError:
        return _extract_example_jpegs(serialized)


def _iter_tfrecord_records(path: str) -> Iterator[bytes]:
    with open(path, "rb") as handle:
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header in {path}")
            length = int.from_bytes(header[:8], "little", signed=False)
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"Truncated TFRecord payload in {path}")
            footer = handle.read(4)
            if len(footer) != 4:
                raise ValueError(f"Truncated TFRecord footer in {path}")
            yield payload


def _sample_lighting(
    *,
    seed: int,
    domains: tuple[str, ...],
    dataset_name: str,
    trajectory_id: int,
    variant_index: int,
    variant_domain_mode: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(
        stable_uint32("train_lighting", seed, dataset_name, int(trajectory_id), int(variant_index))
    )
    if variant_domain_mode == "cycle_domains":
        domain = str(domains[int(variant_index) % len(domains)])
    elif variant_domain_mode == "random":
        domain = str(domains[int(rng.integers(0, len(domains)))])
    else:
        raise ValueError(f"Unknown variant_domain_mode={variant_domain_mode!r}")
    ev_low, ev_high = DEFAULT_EV_RANGES[domain]
    lighting_ev = float(rng.uniform(ev_low, ev_high))
    params = lighting_parameters_for_sample(domain, lighting_ev)
    noise_seed = stable_uint32("train_raw_noise", seed, dataset_name, int(trajectory_id), int(variant_index), domain)
    return {
        "benchmark": "rawvla-bench-light-libero-v1-train-npz",
        "dataset_name": dataset_name,
        "trajectory_id": int(trajectory_id),
        "variant_index": int(variant_index),
        "lighting_domain": domain,
        **params,
        "noise_seed": int(noise_seed),
    }


def _make_episode_raw(
    *,
    image_bytes: np.ndarray,
    wrist_bytes: np.ndarray,
    lighting: dict[str, Any],
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    agentview = []
    wrist = []
    for frame_index, (agent_raw, wrist_raw) in enumerate(zip(image_bytes, wrist_bytes)):
        agent_rgb = _decode_jpeg(bytes(agent_raw))
        wrist_rgb = _decode_jpeg(bytes(wrist_raw))
        agentview.append(
            make_rawvla_observation(
                agent_rgb,
                noise_seed=int(lighting["noise_seed"]),
                frame_index=int(frame_index),
                view_index=0,
                representation=representation,
                sensor_saturation_ev=float(lighting["sensor_saturation_ev"]),
            )
        )
        wrist.append(
            make_rawvla_observation(
                wrist_rgb,
                noise_seed=int(lighting["noise_seed"]),
                frame_index=int(frame_index),
                view_index=1,
                representation=representation,
                sensor_saturation_ev=float(lighting["sensor_saturation_ev"]),
            )
        )
    return np.stack(agentview, axis=0), np.stack(wrist, axis=0)


def _write_npz(path: Path, *, compressed: bool, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **arrays)
        else:
            np.savez(handle, **arrays)
    tmp.replace(path)


def _worker(task_items: list[ShardTask], cfg: dict[str, Any], worker_id: int) -> str:
    out_root = Path(cfg["out_root"])
    domains = tuple(cfg["domains"])
    worker_manifest = out_root / "_manifests" / f"worker_{worker_id:04d}.jsonl"
    worker_manifest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    started = time.time()

    with worker_manifest.open("w", encoding="utf-8") as manifest_f:
        for task in task_items:
            if cfg["max_episodes"] >= 0 and task.episode_offset >= cfg["max_episodes"]:
                continue
            for local_episode_index, serialized in enumerate(_iter_tfrecord_records(task.shard_path)):
                if local_episode_index >= task.episode_count:
                    break
                trajectory_id = task.episode_offset + local_episode_index
                if cfg["max_episodes"] >= 0 and trajectory_id >= cfg["max_episodes"]:
                    continue
                seq = _extract_episode_jpegs(serialized)
                image_bytes = seq["steps/observation/image"]
                wrist_bytes = seq["steps/observation/wrist_image"]
                if len(image_bytes) != len(wrist_bytes):
                    raise ValueError(
                        f"Mismatched camera lengths in {task.rlds_dataset_name} episode {trajectory_id}: "
                        f"image={len(image_bytes)} wrist={len(wrist_bytes)}"
                    )

                for variant_index in range(int(cfg["k_per_episode"])):
                    lighting = _sample_lighting(
                        seed=int(cfg["seed"]),
                        domains=domains,
                        dataset_name=task.lerobot_dataset_name,
                        trajectory_id=trajectory_id,
                        variant_index=variant_index,
                        variant_domain_mode=str(cfg["variant_domain_mode"]),
                    )
                    rel_path = (
                        Path(task.lerobot_dataset_name)
                        / f"episode_{trajectory_id:06d}"
                        / f"variant_{variant_index:02d}.npz"
                    )
                    out_path = out_root / rel_path
                    if out_path.exists() and not cfg["overwrite"]:
                        skipped += 1
                    else:
                        agentview_raw, wrist_raw = _make_episode_raw(
                            image_bytes=image_bytes,
                            wrist_bytes=wrist_bytes,
                            lighting=lighting,
                            representation=str(cfg["representation"]),
                        )
                        metadata = {
                            **lighting,
                            "rlds_dataset_name": task.rlds_dataset_name,
                            "shard_index": task.shard_index,
                            "local_episode_index": local_episode_index,
                            "num_frames": int(agentview_raw.shape[0]),
                            "source_shape": [256, 256, 3],
                            "raw_shape": list(agentview_raw.shape[1:]),
                            "representation": cfg["representation"],
                            "npz_relpath": str(rel_path),
                        }
                        _write_npz(
                            out_path,
                            compressed=bool(cfg["compressed"]),
                            arrays={
                                "agentview_raw_uint8": agentview_raw,
                                "wrist_raw_uint8": wrist_raw,
                                "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
                            },
                        )
                        written += 1
                    manifest_f.write(json.dumps({"npz_relpath": str(rel_path), **lighting}) + "\n")
                    manifest_f.flush()

    return json.dumps(
        {
            "worker_id": worker_id,
            "written": written,
            "skipped": skipped,
            "seconds": round(time.time() - started, 3),
        }
    )


def _discover_shards(data_root: Path) -> list[ShardTask]:
    tasks: list[ShardTask] = []
    for rlds_name, lerobot_name in sorted(RLDS_TO_LEROBOT_NAME.items()):
        dataset_dir = data_root / rlds_name / "1.0.0"
        info_path = dataset_dir / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing dataset_info.json: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        shard_lengths = [int(value) for value in info["splits"][0]["shardLengths"]]
        files = sorted(dataset_dir.glob("*.tfrecord-*"))
        if len(files) != len(shard_lengths):
            raise ValueError(f"{dataset_dir}: expected {len(shard_lengths)} shards, found {len(files)}")
        offset = 0
        for shard_index, (path, count) in enumerate(zip(files, shard_lengths)):
            tasks.append(
                ShardTask(
                    rlds_dataset_name=rlds_name,
                    lerobot_dataset_name=lerobot_name,
                    shard_path=str(path),
                    shard_index=shard_index,
                    episode_offset=offset,
                    episode_count=count,
                )
            )
            offset += count
    return tasks


def _combine_manifests(out_root: Path, cfg: dict[str, Any], worker_results: list[str]) -> None:
    manifest_dir = out_root / "_manifests"
    entries = []
    for path in sorted(manifest_dir.glob("worker_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    entries.sort(key=lambda item: (item["dataset_name"], item["trajectory_id"], item["variant_index"]))
    manifest = {
        "benchmark": "rawvla-bench-light-libero-v1-train-npz",
        "version": 1,
        "created_at_unix": time.time(),
        "config": cfg,
        "ev_ranges": {key: list(value) for key, value in DEFAULT_EV_RANGES.items()},
        "lighting_domains": list(LIGHTING_DOMAINS),
        "unprocess": unprocess_metadata(),
        "worker_results": [json.loads(item) for item in worker_results],
        "num_entries": len(entries),
        "entries": entries,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAWVLA-Light train RAW cache as per-episode NPZ files.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "benchmark_data" / "libero" / "modified_libero_rlds",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "benchmark_data" / "libero" / "rawvla_light_train_cache_npz",
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--k-per-episode", type=int, default=1)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 8, 96))
    parser.add_argument("--max-episodes", type=int, default=-1, help="Per-dataset trajectory-id cap for smoke tests.")
    parser.add_argument("--domains", default=",".join(LIGHTING_DOMAINS))
    parser.add_argument(
        "--variant-domain-mode",
        choices=["random", "cycle_domains"],
        default="random",
        help="random samples a domain per variant; cycle_domains makes variants cover domains in order.",
    )
    parser.add_argument("--representation", choices=["raw", "default_isp"], default="raw")
    parser.add_argument("--compression", choices=["stored", "deflated"], default="stored")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    domains = tuple(part.strip() for part in args.domains.split(",") if part.strip())
    for domain in domains:
        if domain not in LIGHTING_DOMAINS:
            raise ValueError(f"Unknown lighting domain {domain!r}; valid domains are {LIGHTING_DOMAINS}")
    if args.k_per_episode < 1:
        raise ValueError("--k-per-episode must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    tasks = _discover_shards(args.data_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "data_root": str(args.data_root),
        "out_root": str(args.out_root),
        "seed": args.seed,
        "k_per_episode": args.k_per_episode,
        "domains": list(domains),
        "variant_domain_mode": args.variant_domain_mode,
        "representation": args.representation,
        "compressed": args.compression == "deflated",
        "max_episodes": args.max_episodes,
        "overwrite": args.overwrite,
    }
    (args.out_root / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    worker_count = min(args.workers, len(tasks))
    buckets = [[] for _ in range(worker_count)]
    for index, task in enumerate(tasks):
        buckets[index % worker_count].append(task)

    started = time.time()
    if worker_count == 1:
        results = [_worker(buckets[0], cfg, 0)]
    else:
        with mp.get_context("spawn").Pool(worker_count) as pool:
            results = pool.starmap(_worker, [(bucket, cfg, idx) for idx, bucket in enumerate(buckets)])
    _combine_manifests(args.out_root, cfg, results)
    print(json.dumps({"workers": worker_count, "seconds": round(time.time() - started, 3), "results": results}, indent=2))


if __name__ == "__main__":
    main()
