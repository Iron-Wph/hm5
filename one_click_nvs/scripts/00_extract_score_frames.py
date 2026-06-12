from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import ensure_dir, fail, load_config, relpath, write_csv, write_json


def phash_bits(gray: np.ndarray) -> str:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    resized = np.float32(resized)
    dct = cv2.dct(resized)
    block = dct[:8, :8].copy()
    vals = block.flatten()[1:]
    median = float(np.median(vals))
    return "".join("1" if v > median else "0" for v in block.flatten())


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame
    scale = max_width / float(w)
    new_size = (max_width, max(1, int(round(h * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video_path = relpath(cfg, "paths", "video")
    raw_dir = ensure_dir(relpath(cfg, "paths", "frames_raw"))
    splits_dir = ensure_dir(relpath(cfg, "paths", "splits"))

    if not video_path.exists():
        fail(f"Video not found: {video_path}. Put toy.mp4 there first.")

    frame_step = int(cfg["extract"]["frame_step"])
    max_width = int(cfg["extract"]["max_width"])
    jpeg_quality = int(cfg["extract"]["jpeg_quality"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    rows: list[dict[str, object]] = []
    saved_idx = 0
    frame_idx = 0

    pbar = tqdm(total=total if total > 0 else None, desc="Extracting frames")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            frame = resize_max_width(frame, max_width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            name = f"frame_{saved_idx:06d}.jpg"
            out_path = raw_dir / name
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            rows.append(
                {
                    "saved_index": saved_idx,
                    "frame_number": frame_idx,
                    "file_name": name,
                    "path": str(out_path.relative_to(raw_dir.parent.parent)),
                    "blur": laplacian_var(gray),
                    "brightness": float(gray.mean()),
                    "contrast": float(gray.std()),
                    "phash": phash_bits(gray),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                }
            )
            saved_idx += 1
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    quality_csv = splits_dir / "frame_quality.csv"
    write_csv(
        quality_csv,
        rows,
        [
            "saved_index",
            "frame_number",
            "file_name",
            "path",
            "blur",
            "brightness",
            "contrast",
            "phash",
            "width",
            "height",
        ],
    )
    write_json(
        splits_dir / "video_metadata.json",
        {
            "video": str(video_path),
            "source_frame_count": total,
            "fps": fps,
            "frame_step": frame_step,
            "extracted_frame_count": len(rows),
            "raw_frame_dir": str(raw_dir),
        },
    )
    print(f"Extracted {len(rows)} frames to {raw_dir}")
    print(f"Frame quality CSV: {quality_csv}")


if __name__ == "__main__":
    main()
