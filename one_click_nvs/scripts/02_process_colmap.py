from __future__ import annotations

import argparse

from common import ensure_dir, load_config, relpath, require_command, run_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    frames_clean = relpath(cfg, "paths", "frames_clean")
    ns_data = ensure_dir(relpath(cfg, "paths", "nerfstudio_data"))
    matching_method = cfg["colmap"]["matching_method"]

    images = sorted(frames_clean.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No clean frames found in {frames_clean}. Run filtering first.")

    if not args.dry_run:
        require_command("ns-process-data")
        require_command("ffmpeg")

    cmd = [
        "ns-process-data",
        "images",
        "--data",
        str(frames_clean),
        "--output-dir",
        str(ns_data),
        "--matching-method",
        str(matching_method),
    ]
    run_command(cmd, dry_run=args.dry_run)

    print(f"Nerfstudio dataset directory: {ns_data}")
    print("Expected key output: transforms.json")


if __name__ == "__main__":
    main()
