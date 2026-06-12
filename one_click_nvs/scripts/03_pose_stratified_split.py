from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

from common import ensure_dir, load_config, read_json, relpath, write_csv, write_json


def camera_center(frame: dict) -> tuple[float, float, float]:
    matrix = frame.get("transform_matrix")
    if not matrix:
        return 0.0, 0.0, 0.0
    return float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])


def assign_split(index: int) -> str:
    pattern = ["train", "train", "train", "val", "test"]
    return pattern[index % len(pattern)]


def make_prefixed_frame(frame: dict, ns_data: Path, split_images: Path, split: str) -> dict:
    original_rel = Path(frame["file_path"])
    original = ns_data / original_rel
    prefix = "train" if split == "train" else f"eval_{split}"
    target_rel = Path("images_split") / f"{prefix}_{original_rel.name}"
    target = ns_data / target_rel
    ensure_dir(target.parent)
    shutil.copy2(original, target)
    new_frame = copy.deepcopy(frame)
    new_frame["file_path"] = target_rel.as_posix()
    new_frame["split"] = split
    return new_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ns_data = relpath(cfg, "paths", "nerfstudio_data")
    splits_dir = ensure_dir(relpath(cfg, "paths", "splits"))
    transforms_path = ns_data / "transforms.json"
    split_images = ensure_dir(ns_data / "images_split")

    if not transforms_path.exists():
        raise SystemExit(f"Missing {transforms_path}. Run ns-process-data first.")

    meta = read_json(transforms_path)
    frames = list(meta.get("frames", []))
    if not frames:
        raise SystemExit(f"No frames found in {transforms_path}")
    if len(frames) < 20:
        raise SystemExit(
            f"Only {len(frames)} frame(s) were registered in {transforms_path}. "
            "This is too few for a valid NeRF/3DGS experiment. "
            "Rerun COLMAP/Nerfstudio data processing with more successful registrations before training."
        )

    frames = sorted(frames, key=lambda f: Path(f["file_path"]).name)
    split_rows: list[dict[str, object]] = []
    all_split_frames: list[dict] = []
    split_filenames: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for i, frame in enumerate(frames):
        split = assign_split(i)
        x, y, z = camera_center(frame)
        prefixed = make_prefixed_frame(frame, ns_data, split_images, split)
        all_split_frames.append(prefixed)
        split_filenames[split].append(prefixed["file_path"])

        split_rows.append(
            {
                "index": i,
                "original_file_path": frame["file_path"],
                "split_file_path": prefixed["file_path"],
                "split": split,
                "x": x,
                "y": y,
                "z": z,
            }
        )

    counts = {key: len(value) for key, value in split_filenames.items()}
    if counts["train"] == 0 or counts["val"] == 0 or counts["test"] == 0:
        raise SystemExit(
            f"Invalid split counts: {counts}. "
            "Need non-empty train/val/test splits before training and evaluation."
        )

    split_meta = copy.deepcopy(meta)
    split_meta["frames"] = all_split_frames
    split_meta["train_filenames"] = split_filenames["train"]
    split_meta["val_filenames"] = split_filenames["val"]
    split_meta["test_filenames"] = split_filenames["test"]
    split_meta["split_strategy"] = cfg["split"]["strategy"]
    write_json(ns_data / "transforms_splits.json", split_meta)

    # Backward-compatible aliases for older instructions/scripts.
    write_json(ns_data / "transforms_all_split.json", split_meta)
    write_json(ns_data / "transforms_trainval.json", split_meta)
    write_json(ns_data / "transforms_traintest.json", split_meta)
    write_csv(
        splits_dir / "split.csv",
        split_rows,
        ["index", "original_file_path", "split_file_path", "split", "x", "y", "z"],
    )

    write_json(splits_dir / "split_summary.json", counts)
    print(f"Split counts: {counts}")
    print(f"Split JSON: {ns_data / 'transforms_splits.json'}")
    print("Default training uses explicit train_filenames/val_filenames/test_filenames.")


if __name__ == "__main__":
    main()
