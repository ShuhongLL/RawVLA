"""Fit the two fixed default-ISP profiles from the three ColorChecker images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

CALIBRATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CALIBRATION_DIR))

from default_isp import apply_default_isp, linear_to_srgb  # noqa: E402


# Nominal ColorChecker Classic sRGB values, row-major in the standard chart orientation.
COLORCHECKER_SRGB8 = np.asarray(
    [
        [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
        [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
        [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
        [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
    ],
    dtype=np.float64,
) / 255.0

# The head chart is very small and rotated 180 degrees, so its centers are
# explicitly recorded rather than inferred from an unstable corner detector.
AGENT_PATCH_CENTERS = np.asarray(
    [
        [(88, 58), (94, 58), (101, 58), (107, 58), (113, 58), (120, 58)],
        [(88, 63), (95, 63), (101, 63), (107, 63), (114, 63), (120, 63)],
        [(88, 69), (95, 69), (101, 69), (108, 69), (114, 69), (121, 69)],
        [(88, 74), (95, 74), (101, 74), (108, 74), (115, 74), (122, 74)],
    ],
    dtype=np.float64,
).reshape(-1, 2)

# TL, TR, BR, BL centers of the four corner patches. Interior centers are
# obtained with bilinear interpolation, which handles perspective sufficiently
# well at the wrist images' resolution.
WRIST_CORNER_CENTERS = {
    "left_wrist": np.asarray([(112, 58), (201, 43), (196, 94), (119, 107)], dtype=np.float64),
    "right_wrist": np.asarray([(33, 37), (127, 44), (123, 95), (42, 89)], dtype=np.float64),
}


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.asarray(srgb, dtype=np.float64)
    return np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4))


def grid_centers(corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    points = []
    for row in range(4):
        v = row / 3.0
        left = (1.0 - v) * tl + v * bl
        right = (1.0 - v) * tr + v * br
        for column in range(6):
            u = column / 5.0
            points.append((1.0 - u) * left + u * right)
    return np.asarray(points)


def sample_patches(image: np.ndarray, centers: np.ndarray) -> np.ndarray:
    samples = []
    for x_float, y_float in centers:
        x, y = int(round(x_float)), int(round(y_float))
        samples.append(np.median(image[y - 1 : y + 2, x - 1 : x + 2], axis=(0, 1)))
    return np.asarray(samples, dtype=np.float64) / 255.0


def decode_parameters(parameters: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    exposure = float(np.exp(parameters[0]))
    white_balance = np.asarray([np.exp(parameters[1]), 1.0, np.exp(parameters[2])])
    off_diagonal = parameters[3:].reshape(3, 2)
    matrix = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        other_columns = [column for column in range(3) if column != row]
        matrix[row, other_columns] = off_diagonal[row]
        matrix[row, row] = 1.0 - off_diagonal[row].sum()
    return exposure, white_balance, matrix


def fit_profile(
    sample_sets: list[np.ndarray],
    targets: list[np.ndarray],
    *,
    neutral_at_start: bool,
) -> dict[str, object]:
    samples = np.concatenate(sample_sets)
    target = np.concatenate(targets)
    patch_weights = (
        np.r_[np.full(6, 1.5), np.ones(18)]
        if neutral_at_start
        else np.r_[np.ones(18), np.full(6, 1.5)]
    )
    weights = np.tile(patch_weights, len(sample_sets))[:, None]

    def residual(parameters: np.ndarray) -> np.ndarray:
        exposure, white_balance, matrix = decode_parameters(parameters)
        prediction = (samples * white_balance * exposure) @ matrix.T
        color_error = ((prediction - target) * weights).ravel()
        matrix_regularizer = 0.05 * (matrix - np.eye(3)).ravel()
        wb_regularizer = 0.01 * parameters[1:3]
        return np.concatenate((color_error, matrix_regularizer, wb_regularizer))

    result = least_squares(residual, np.zeros(9), loss="soft_l1", f_scale=0.03, max_nfev=10_000)
    if not result.success:
        raise RuntimeError(f"Color calibration did not converge: {result.message}")
    exposure, white_balance, matrix = decode_parameters(result.x)
    return {
        "black_level": [0.0, 0.0, 0.0],
        "exposure_gain": exposure,
        "exposure_ev": float(np.log2(exposure)),
        "white_balance": white_balance.tolist(),
        "color_matrix": matrix.tolist(),
    }


def make_preview(source: Image.Image, corrected: np.ndarray, title: str) -> Image.Image:
    width, height = source.size
    canvas = Image.new("RGB", (2 * width, height + 22), "white")
    canvas.paste(source, (0, 22))
    canvas.paste(Image.fromarray(corrected), (width, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"{title}: RAW", fill="black")
    draw.text((width + 4, 4), "fixed default ISP", fill="black")
    return canvas


def patch_mae8(samples: np.ndarray, target: np.ndarray, profile: dict[str, object]) -> float:
    exposure = float(profile["exposure_gain"])
    wb = np.asarray(profile["white_balance"])
    matrix = np.asarray(profile["color_matrix"])
    prediction = np.clip(linear_to_srgb((samples * wb * exposure) @ matrix.T), 0.0, 1.0)
    target_srgb = np.clip(linear_to_srgb(target), 0.0, 1.0)
    return float(np.abs(prediction - target_srgb).mean() * 255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=CALIBRATION_DIR / "input")
    parser.add_argument("--output-config", type=Path, default=CALIBRATION_DIR / "default_isp.json")
    parser.add_argument("--preview-dir", type=Path, default=CALIBRATION_DIR / "preview")
    parser.add_argument("--report", type=Path, default=CALIBRATION_DIR / "calibration_report.json")
    args = parser.parse_args()

    images = {
        "agentview": np.asarray(Image.open(args.input_dir / "head_camera.png").convert("RGB")),
        "left_wrist": np.asarray(Image.open(args.input_dir / "left_wrist.png").convert("RGB")),
        "right_wrist": np.asarray(Image.open(args.input_dir / "right_wrist.png").convert("RGB")),
    }
    samples = {
        "agentview": sample_patches(images["agentview"], AGENT_PATCH_CENTERS),
        "left_wrist": sample_patches(images["left_wrist"], grid_centers(WRIST_CORNER_CENTERS["left_wrist"])),
        "right_wrist": sample_patches(images["right_wrist"], grid_centers(WRIST_CORNER_CENTERS["right_wrist"])),
    }
    target = srgb_to_linear(COLORCHECKER_SRGB8)
    agent_target = target[::-1]  # chart is rotated 180 degrees in head_camera.png

    agent_profile = fit_profile([samples["agentview"]], [agent_target], neutral_at_start=True)
    wrist_profile = fit_profile(
        [samples["left_wrist"], samples["right_wrist"]],
        [target, target],
        neutral_at_start=False,
    )
    agent_profile["source_image"] = "input/head_camera.png"
    wrist_profile["source_images"] = ["input/left_wrist.png", "input/right_wrist.png"]
    wrist_profile["shared_by"] = ["left_wrist", "right_wrist"]
    config = {
        "schema_version": 1,
        "calibration": "ColorChecker Classic 24-patch nominal sRGB/D65",
        "input_domain": "three_channel_linear_raw",
        "tone_curve": "linear_to_srgb",
        "profiles": {"agentview": agent_profile, "wrist": wrist_profile},
    }

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    metrics = {
        "agentview_patch_rgb_mae_8bit": patch_mae8(samples["agentview"], agent_target, agent_profile),
        "left_wrist_patch_rgb_mae_8bit": patch_mae8(samples["left_wrist"], target, wrist_profile),
        "right_wrist_patch_rgb_mae_8bit": patch_mae8(samples["right_wrist"], target, wrist_profile),
    }
    args.report.write_text(json.dumps({"metrics": metrics, "config": config}, indent=2) + "\n", encoding="utf-8")

    args.preview_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        corrected = apply_default_isp(image, name, config=config)
        source = Image.fromarray(image)
        make_preview(source, corrected, name).save(args.preview_dir / f"{name}_raw_vs_default_isp.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
