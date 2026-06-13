from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from common import ensure_dir, load_config, relpath, write_csv, write_json


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def look_at_c2w(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    # Nerfstudio expects camera-to-world matrices in OpenGL-style coordinates:
    # camera looks along local -Z, local +Y is up.
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def assign_split(index: int) -> str:
    pattern = ["train", "train", "train", "val", "test"]
    return pattern[index % len(pattern)]


def write_sparse_ply(path: Path, count: int, radius: float) -> None:
    rng = np.random.default_rng(42)
    points = rng.normal(size=(count, 3)).astype(np.float32)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    scales = rng.uniform(0.05, radius * 0.35, size=(count, 1)).astype(np.float32)
    points = points * scales
    colors = np.full((count, 3), 180, dtype=np.uint8)
    ensure_dir(path.parent)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def crop_bounds(width: int, height: int, focus_cfg: dict) -> tuple[int, int, int, int]:
    if not bool(focus_cfg.get("enabled", False)):
        return 0, 0, width, height
    crop_w = max(32, int(round(width * float(focus_cfg.get("width_fraction", 1.0)))))
    crop_h = max(32, int(round(height * float(focus_cfg.get("height_fraction", 1.0)))))
    cx = int(round(width * float(focus_cfg.get("center_x", 0.5))))
    cy = int(round(height * float(focus_cfg.get("center_y", 0.5))))
    x0 = max(0, min(width - crop_w, cx - crop_w // 2))
    y0 = max(0, min(height - crop_h, cy - crop_h // 2))
    return x0, y0, x0 + crop_w, y0 + crop_h


def crop_image(image: np.ndarray, focus_cfg: dict) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = crop_bounds(w, h, focus_cfg)
    return image[y0:y1, x0:x1].copy()


def make_focus_mask(width: int, height: int, focus_cfg: dict) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not bool(focus_cfg.get("mask_enabled", False)):
        mask[:, :] = 255
        return mask
    center = (
        int(round(width * float(focus_cfg.get("mask_center_x", 0.5)))),
        int(round(height * float(focus_cfg.get("mask_center_y", 0.5)))),
    )
    axes = (
        max(1, int(round(width * float(focus_cfg.get("mask_radius_x", 0.48))))),
        max(1, int(round(height * float(focus_cfg.get("mask_radius_y", 0.48))))),
    )
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_turntable.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    frames_dir = relpath(cfg, "paths", "frames_clean")
    out_dir = ensure_dir(relpath(cfg, "paths", "nerfstudio_data"))
    image_dir = ensure_dir(out_dir / "images")
    mask_dir = ensure_dir(out_dir / "masks")
    splits_dir = ensure_dir(relpath(cfg, "paths", "splits"))
    focus_cfg = cfg.get("object_focus", {})

    image_paths = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )
    if not image_paths:
        raise SystemExit(f"No clean frames found in {frames_dir}. Run extraction/filtering first.")

    max_frames = int(cfg["turntable"].get("max_frames", 0))
    if max_frames > 0 and len(image_paths) > max_frames:
        indices = np.linspace(0, len(image_paths) - 1, max_frames, dtype=int)
        image_paths = [image_paths[i] for i in indices]

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise SystemExit(f"Could not read image: {image_paths[0]}")
    first = crop_image(first, focus_cfg)
    height, width = first.shape[:2]
    fov = math.radians(float(cfg["turntable"]["fov_degrees"]))
    fl = 0.5 * width / math.tan(0.5 * fov)
    cx = width / 2.0
    cy = height / 2.0
    radius = float(cfg["turntable"]["radius"])
    elevation = math.radians(float(cfg["turntable"]["elevation_degrees"]))
    total_angle = math.radians(float(cfg["turntable"]["total_angle_degrees"]))
    start_angle = math.radians(float(cfg["turntable"]["start_angle_degrees"]))

    frames: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    split_filenames: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    train_all_frames = bool(cfg.get("split", {}).get("train_all_frames", False))

    for i, src in enumerate(image_paths):
        theta = start_angle + total_angle * i / max(1, len(image_paths))
        eye = np.array(
            [
                radius * math.sin(theta) * math.cos(elevation),
                radius * math.sin(elevation),
                radius * math.cos(theta) * math.cos(elevation),
            ],
            dtype=np.float32,
        )
        c2w = look_at_c2w(eye, np.zeros(3, dtype=np.float32), np.array([0, 1, 0], dtype=np.float32))
        split = assign_split(i)
        dst_name = f"{split}_{i:06d}{src.suffix.lower()}"
        dst_rel = Path("images") / dst_name
        image = cv2.imread(str(src))
        if image is None:
            continue
        image = crop_image(image, focus_cfg)
        cv2.imwrite(str(out_dir / dst_rel), image)
        mask_rel = Path("masks") / f"{split}_{i:06d}.png"
        cv2.imwrite(str(out_dir / mask_rel), make_focus_mask(image.shape[1], image.shape[0], focus_cfg))
        frame = {
            "file_path": dst_rel.as_posix(),
            "mask_path": mask_rel.as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)
        if train_all_frames:
            split_filenames["train"].append(dst_rel.as_posix())
            if split in {"val", "test"}:
                split_filenames[split].append(dst_rel.as_posix())
        else:
            split_filenames[split].append(dst_rel.as_posix())
        split_rows.append(
            {
                "index": i,
                "source": str(src),
                "file_path": dst_rel.as_posix(),
                "split": split,
                "theta_degrees": math.degrees(theta),
                "x": float(eye[0]),
                "y": float(eye[1]),
                "z": float(eye[2]),
            }
        )

    if any(len(v) == 0 for v in split_filenames.values()):
        raise SystemExit(f"Invalid split counts: { {k: len(v) for k, v in split_filenames.items()} }")

    meta: dict[str, object] = {
        "camera_model": "OPENCV",
        "fl_x": fl,
        "fl_y": fl,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "orientation_override": "none",
        "frames": frames,
        "train_filenames": split_filenames["train"],
        "val_filenames": split_filenames["val"],
        "test_filenames": split_filenames["test"],
        "turntable_pose_prior": cfg["turntable"],
        "object_focus": focus_cfg,
        "train_all_frames": train_all_frames,
    }

    if bool(cfg["turntable"].get("generate_sparse_ply", True)):
        ply_name = "sparse_pc.ply"
        write_sparse_ply(out_dir / ply_name, int(cfg["turntable"].get("sparse_points", 8000)), radius)
        meta["ply_file_path"] = ply_name

    write_json(out_dir / "transforms.json", meta)
    write_csv(
        splits_dir / "split.csv",
        split_rows,
        ["index", "source", "file_path", "split", "theta_degrees", "x", "y", "z"],
    )
    write_json(splits_dir / "turntable_pose_summary.json", {key: len(value) for key, value in split_filenames.items()})

    print(f"Turntable dataset written to: {out_dir}")
    print(f"Frames: {len(frames)}")
    print(f"Split counts: { {key: len(value) for key, value in split_filenames.items()} }")
    print(f"Focal length estimate: {fl:.2f}px, image size: {width}x{height}")


if __name__ == "__main__":
    main()
