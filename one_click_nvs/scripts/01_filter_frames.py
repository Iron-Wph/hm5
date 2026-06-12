from __future__ import annotations

import argparse
import shutil

import numpy as np

from common import ensure_dir, hamming_bits, load_config, read_csv, relpath, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = relpath(cfg, "paths", "frames_raw")
    clean_dir = ensure_dir(relpath(cfg, "paths", "frames_clean"))
    splits_dir = ensure_dir(relpath(cfg, "paths", "splits"))
    quality_csv = splits_dir / "frame_quality.csv"

    rows = read_csv(quality_csv)
    if not rows:
        raise SystemExit("No rows in frame_quality.csv. Run 00_extract_score_frames.py first.")

    blur_values = np.array([float(r["blur"]) for r in rows], dtype=np.float64)
    blur_threshold = float(np.quantile(blur_values, float(cfg["filter"]["blur_quantile"])))
    brightness_min = float(cfg["filter"]["brightness_min"])
    brightness_max = float(cfg["filter"]["brightness_max"])
    contrast_min = float(cfg["filter"]["contrast_min"])
    min_hash_dist = int(cfg["filter"]["phash_min_distance"])
    max_frames = int(cfg["filter"]["max_frames"])

    rows = sorted(rows, key=lambda r: int(r["frame_number"]))
    kept: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    last_hash: str | None = None

    for r in rows:
        reasons: list[str] = []
        blur = float(r["blur"])
        brightness = float(r["brightness"])
        contrast = float(r["contrast"])
        current_hash = r["phash"]

        if blur < blur_threshold:
            reasons.append("blur")
        if brightness < brightness_min or brightness > brightness_max:
            reasons.append("brightness")
        if contrast < contrast_min:
            reasons.append("contrast")
        if last_hash is not None and hamming_bits(last_hash, current_hash) < min_hash_dist:
            reasons.append("duplicate")

        if reasons:
            item = dict(r)
            item["keep"] = False
            item["reject_reason"] = ";".join(reasons)
            rejected.append(item)
            continue

        item = dict(r)
        item["keep"] = True
        item["reject_reason"] = ""
        kept.append(item)
        last_hash = current_hash

    if max_frames > 0 and len(kept) > max_frames:
        selected_indices = set(np.linspace(0, len(kept) - 1, max_frames, dtype=int).tolist())
        reduced = [r for i, r in enumerate(kept) if i in selected_indices]
        for i, r in enumerate(kept):
            if i not in selected_indices:
                item = dict(r)
                item["keep"] = False
                item["reject_reason"] = "max_frames_subsample"
                rejected.append(item)
        kept = reduced

    clean_rows: list[dict[str, object]] = []
    for new_idx, r in enumerate(kept):
        src = raw_dir / str(r["file_name"])
        dst_name = f"clean_{new_idx:06d}.jpg"
        dst = clean_dir / dst_name
        shutil.copy2(src, dst)
        out = dict(r)
        out["clean_index"] = new_idx
        out["clean_file_name"] = dst_name
        out["clean_path"] = str(dst)
        clean_rows.append(out)

    write_csv(
        splits_dir / "clean_frame_list.csv",
        clean_rows,
        [
            "clean_index",
            "saved_index",
            "frame_number",
            "file_name",
            "path",
            "clean_file_name",
            "blur",
            "brightness",
            "contrast",
            "phash",
            "width",
            "height",
            "clean_path",
            "keep",
            "reject_reason",
        ],
    )
    write_csv(
        splits_dir / "rejected_frame_list.csv",
        rejected,
        list(rejected[0].keys()) if rejected else ["file_name", "reject_reason"],
    )
    write_json(
        splits_dir / "filter_summary.json",
        {
            "input_frames": len(rows),
            "kept_frames": len(kept),
            "rejected_frames": len(rejected),
            "blur_threshold": blur_threshold,
            "brightness_range": [brightness_min, brightness_max],
            "contrast_min": contrast_min,
            "phash_min_distance": min_hash_dist,
            "max_frames": max_frames,
        },
    )
    print(f"Kept {len(kept)} frames in {clean_dir}")
    print(f"Rejected {len(rejected)} frames")


if __name__ == "__main__":
    main()
