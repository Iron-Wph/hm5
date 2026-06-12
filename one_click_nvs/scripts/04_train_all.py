from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import ensure_dir, load_config, newest_file, relpath, require_command, run_command, write_json


def find_existing_config(outputs: Path, experiment_name: str, method: str) -> Path | None:
    run_root = outputs / experiment_name / method
    if not run_root.exists():
        return None
    configs = list(run_root.glob("*/config.yml"))
    if not configs:
        return None
    return max(configs, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs = ensure_dir(relpath(cfg, "paths", "outputs"))
    results = ensure_dir(relpath(cfg, "paths", "results"))
    dataset_json = relpath(cfg, "training", "dataset_json")
    max_iter = str(cfg["training"]["max_num_iterations"])

    if not dataset_json.exists():
        raise SystemExit(f"Missing dataset JSON: {dataset_json}. Run split step first.")
    if not args.dry_run:
        require_command("ns-train")

    manifest: dict[str, object] = {"dataset_json": str(dataset_json), "models": []}

    for model in cfg["training"]["models"]:
        name = model["name"]
        method = model["method"]
        model_args = list(model.get("args", []))
        existing_config = find_existing_config(outputs, name, method)
        if existing_config and not args.force_retrain:
            print(f"Skipping {name}; existing config found: {existing_config}")
            manifest["models"].append(
                {
                    "name": name,
                    "method": method,
                    "args": model_args,
                    "train_seconds": None,
                    "config_path": str(existing_config),
                    "skipped_existing": True,
                }
            )
            continue

        start = time.time()
        cmd = [
            "ns-train",
            method,
            "--experiment-name",
            name,
            "--output-dir",
            str(outputs),
            "--viewer.quit-on-train-completion",
            "True",
            "--max-num-iterations",
            max_iter,
            "--data",
            str(dataset_json),
            *model_args,
            "nerfstudio-data",
            "--eval-mode",
            "filename",
        ]
        run_command(cmd, dry_run=args.dry_run)
        elapsed = time.time() - start
        config_path = None if args.dry_run else newest_file(outputs, "config.yml", min_mtime=start - 5)
        manifest["models"].append(
            {
                "name": name,
                "method": method,
                "args": model_args,
                "train_seconds": elapsed,
                "config_path": str(config_path) if config_path else None,
            }
        )
        print(f"Finished {name}. config={config_path}")

    write_json(results / "run_manifest.json", manifest)
    print(f"Run manifest: {results / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
