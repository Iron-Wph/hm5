from __future__ import annotations

import argparse
import csv
import gc
import subprocess
from pathlib import Path

import numpy as np

from common import ensure_dir, flatten_numeric, load_config, read_json, relpath, require_command, run_command, write_csv, write_json


def metric_from_flat(flat: dict[str, float], token: str, prefer: str = "mean") -> float | None:
    candidates = {k: v for k, v in flat.items() if token.lower() in k.lower()}
    if not candidates:
        return None
    preferred = {k: v for k, v in candidates.items() if prefer.lower() in k.lower()}
    if preferred:
        return list(preferred.values())[0]
    return list(candidates.values())[0]


def choose_metric_tag(tags: list[str], token: str) -> str | None:
    token = token.lower()
    candidates = [tag for tag in tags if token in tag.lower()]
    if not candidates:
        return None

    def score(tag: str) -> tuple[int, str]:
        lower = tag.lower()
        value = 0
        if "eval" in lower:
            value += 20
        if "all" in lower:
            value += 8
        if "image" in lower:
            value += 5
        if "train" in lower:
            value -= 10
        if "loss" in lower:
            value -= 20
        return value, tag

    return max(candidates, key=score)


def load_logged_metrics(config_path: str | None) -> tuple[dict[str, float], dict[str, str]]:
    if not config_path:
        return {}, {}
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return {}, {}

    run_dir = Path(config_path).parent
    event_files = sorted(run_dir.rglob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not event_files:
        return {}, {}

    best_values: dict[str, tuple[int, float]] = {}
    best_tags: dict[str, str] = {}

    for event_file in event_files:
        try:
            accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = accumulator.Tags().get("scalars", [])
        except Exception:
            continue

        for metric in ["psnr", "ssim", "lpips"]:
            tag = choose_metric_tag(tags, metric)
            if not tag:
                continue
            values = accumulator.Scalars(tag)
            if not values:
                continue
            latest = max(values, key=lambda item: item.step)
            current = best_values.get(metric)
            if current is None or latest.step >= current[0]:
                best_values[metric] = (int(latest.step), float(latest.value))
                best_tags[metric] = f"{event_file.name}:{tag}"

    return {metric: value for metric, (_step, value) in best_values.items()}, best_tags


def dataloader_length(loader: object) -> int:
    try:
        return int(len(loader))  # type: ignore[arg-type]
    except Exception:
        return -1


def as_float(value: object) -> float:
    try:
        import torch

        if torch.is_tensor(value):
            return float(value.detach().float().mean().cpu())
    except Exception:
        pass
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def to_device(value: object, device: object) -> object:
    try:
        import torch

        if torch.is_tensor(value):
            return value.to(device)
    except Exception:
        pass
    if isinstance(value, dict):
        return {key: to_device(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [to_device(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(to_device(child, device) for child in value)
    return value


def average_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = sorted({key for row in metric_dicts for key in row})
    out: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in metric_dicts if key in row and not np.isnan(row[key])]
        if values:
            out[key] = float(np.mean(values))
            if len(values) > 1:
                out[f"{key}_std"] = float(np.std(values))
    return out


def manual_eval_items(pipeline: object, items: list[tuple[object, dict]], output_path: Path, prefix: str) -> dict[str, float]:
    if not items:
        return {}

    import torch
    import torchvision.utils as vutils

    device = pipeline.device  # type: ignore[attr-defined]
    pipeline.eval()  # type: ignore[attr-defined]
    metric_rows: list[dict[str, float]] = []
    output_path.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        for idx, (camera, batch) in enumerate(items):
            camera = camera.to(device)  # type: ignore[attr-defined]
            batch = to_device(batch, device)
            outputs = pipeline.model.get_outputs_for_camera(camera)  # type: ignore[attr-defined]
            metrics_dict, images_dict = pipeline.model.get_image_metrics_and_images(outputs, batch)  # type: ignore[attr-defined]
            metric_rows.append({key: as_float(value) for key, value in metrics_dict.items()})
            if idx < 6:
                for key, image in images_dict.items():
                    try:
                        if torch.is_tensor(image):
                            vutils.save_image(
                                image.detach().permute(2, 0, 1).cpu(),
                                output_path / f"{prefix}_{idx:04d}_{key}.png",
                            )
                    except Exception:
                        pass
    pipeline.train()  # type: ignore[attr-defined]
    return average_metric_dicts(metric_rows)


def items_from_loader(loader: object) -> list[tuple[object, dict]]:
    items: list[tuple[object, dict]] = []
    try:
        for item in loader:  # type: ignore[operator]
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                items.append((item[0], item[1]))
            elif isinstance(item, list) and item and isinstance(item[0], tuple) and len(item[0]) == 2:
                camera, batch = item[0]
                if isinstance(batch, dict):
                    items.append((camera, batch))
    except Exception:
        return []
    return items


def items_from_dataset(datamanager: object) -> list[tuple[object, dict]]:
    try:
        dataset = datamanager.eval_dataset  # type: ignore[attr-defined]
        cameras = dataset.cameras  # type: ignore[attr-defined]
        items = []
        for idx in range(len(dataset)):
            camera = cameras[idx : idx + 1]
            batch = dataset.get_data(idx)
            items.append((camera, batch))
        return items
    except Exception:
        return []


def run_fallback_pipeline_eval(config_path: str | None, output_path: Path) -> tuple[dict[str, float], str]:
    if not config_path:
        return {}, "missing_config"

    try:
        import torch
        from nerfstudio.utils.eval_utils import eval_setup
    except Exception as exc:
        return {}, f"fallback_import_failed: {exc}"

    try:
        _config, pipeline, _checkpoint_path, _step = eval_setup(Path(config_path), test_mode="test")
        candidates: list[tuple[str, object]] = []
        datamanager = pipeline.datamanager
        for attr in ["fixed_indices_eval_dataloader", "eval_dataloader", "fixed_indices_train_dataloader", "train_dataloader"]:
            if hasattr(datamanager, attr):
                candidates.append((attr, getattr(datamanager, attr)))

        for loader_name, loader in candidates:
            length = dataloader_length(loader)
            if length == 0:
                continue
            if not hasattr(pipeline, "get_average_image_metrics"):
                continue
            try:
                metrics = pipeline.get_average_image_metrics(
                    loader,
                    image_prefix=loader_name.replace("_dataloader", ""),
                    output_path=output_path,
                    get_std=True,
                )
                if metrics:
                    return {key: float(value) for key, value in metrics.items()}, f"fallback_pipeline:{loader_name}"
            except Exception:
                pass

            items = items_from_loader(loader)
            metrics = manual_eval_items(pipeline, items, output_path, f"manual_{loader_name}")
            if metrics:
                return metrics, f"manual_loader:{loader_name}"

        items = items_from_dataset(datamanager)
        metrics = manual_eval_items(pipeline, items, output_path, "manual_eval_dataset")
        if metrics:
            return metrics, "manual_eval_dataset"

        return {}, "fallback_no_nonempty_dataloader"
    except Exception as exc:
        return {}, f"fallback_failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            if "pipeline" in locals() and hasattr(pipeline, "datamanager"):
                datamanager = pipeline.datamanager
                for attr in ["train_dataparser_outputs", "eval_dataparser_outputs", "train_dataset", "eval_dataset"]:
                    try:
                        setattr(datamanager, attr, None)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            del pipeline  # type: ignore[name-defined]
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()  # type: ignore[name-defined]
        except Exception:
            pass


def load_image(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def compute_pair_metrics(pred: Path, gt: Path) -> dict[str, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    pred_img = load_image(pred)
    gt_img = load_image(gt)
    if pred_img.shape != gt_img.shape:
        return {}
    psnr = float(peak_signal_noise_ratio(gt_img, pred_img, data_range=1.0))
    ssim = float(structural_similarity(gt_img, pred_img, channel_axis=2, data_range=1.0))
    return {"custom_psnr": psnr, "custom_ssim": ssim}


def discover_pairs(render_dir: Path) -> list[tuple[Path, Path]]:
    images = [p for p in render_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    gt = [p for p in images if any(t in p.as_posix().lower() for t in ["gt", "ground_truth", "target"])]
    pred = [p for p in images if any(t in p.as_posix().lower() for t in ["render", "rgb", "prediction"])]
    pairs: list[tuple[Path, Path]] = []
    for i, g in enumerate(sorted(gt)):
        if i < len(pred):
            pairs.append((sorted(pred)[i], g))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-ns-eval", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = ensure_dir(relpath(cfg, "paths", "results"))
    tables = ensure_dir(relpath(cfg, "paths", "tables"))
    manifest_path = results / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing run manifest: {manifest_path}. Run training first.")

    manifest = read_json(manifest_path)
    run_ns_eval = bool(cfg["evaluation"]["run_ns_eval"]) and not args.skip_ns_eval
    if run_ns_eval and not args.dry_run:
        require_command("ns-eval")

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for model in manifest.get("models", []):
        name = model["name"]
        config_path = model.get("config_path")
        model_dir = ensure_dir(results / name)
        eval_json = model_dir / "eval.json"
        render_dir = ensure_dir(model_dir / "renders")

        if config_path and run_ns_eval:
            cmd = [
                "ns-eval",
                "--load-config",
                str(config_path),
                "--output-path",
                str(eval_json),
                "--render-output-path",
                str(render_dir),
            ]
            try:
                run_command(cmd, dry_run=args.dry_run)
                eval_error = ""
            except subprocess.CalledProcessError as exc:
                eval_error = f"ns-eval failed with exit code {exc.returncode}"
                print(f"WARNING: {name}: {eval_error}")
        else:
            eval_error = ""

        flat: dict[str, float] = {}
        if eval_json.exists():
            flat = dict(flatten_numeric("", read_json(eval_json)))
            for key, value in flat.items():
                detail_rows.append({"model": name, "metric": key, "value": value})

        fallback_error = ""
        fallback_source = ""
        if not any(metric_from_flat(flat, token) is not None for token in ["psnr", "ssim", "lpips"]):
            fallback_json = model_dir / "eval_fallback.json"
            fallback_metrics, fallback_source = run_fallback_pipeline_eval(config_path, render_dir)
            if fallback_metrics:
                write_json(fallback_json, {"metrics": fallback_metrics, "source": fallback_source})
                flat.update(fallback_metrics)
                for key, value in fallback_metrics.items():
                    detail_rows.append({"model": name, "metric": f"{fallback_source}.{key}", "value": value})
            else:
                fallback_error = fallback_source

        custom_values: list[dict[str, float]] = []
        if cfg["evaluation"]["compute_custom_metrics_when_pairs_exist"] and render_dir.exists():
            for pred, gt in discover_pairs(render_dir):
                metrics = compute_pair_metrics(pred, gt)
                if metrics:
                    custom_values.append(metrics)

        custom_psnr = None
        custom_ssim = None
        if custom_values:
            custom_psnr = float(np.mean([m["custom_psnr"] for m in custom_values]))
            custom_ssim = float(np.mean([m["custom_ssim"] for m in custom_values]))

        logged_metrics, logged_tags = load_logged_metrics(config_path)
        psnr = metric_from_flat(flat, "psnr")
        ssim = metric_from_flat(flat, "ssim")
        lpips = metric_from_flat(flat, "lpips")
        metric_source = "ns-eval" if any(v is not None for v in [psnr, ssim, lpips]) else ""
        if fallback_source.startswith("fallback_pipeline") and any(v is not None for v in [psnr, ssim, lpips]):
            metric_source = fallback_source
        if psnr is None and "psnr" in logged_metrics:
            psnr = logged_metrics["psnr"]
            metric_source = "training_log"
        if ssim is None and "ssim" in logged_metrics:
            ssim = logged_metrics["ssim"]
            metric_source = "training_log"
        if lpips is None and "lpips" in logged_metrics:
            lpips = logged_metrics["lpips"]
            metric_source = "training_log"

        row = {
            "model": name,
            "method": model.get("method"),
            "train_seconds": model.get("train_seconds"),
            "config_path": config_path,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "custom_psnr": custom_psnr,
            "custom_ssim": custom_ssim,
            "metric_source": metric_source,
            "metric_log_tags": "; ".join(f"{key}={value}" for key, value in sorted(logged_tags.items())),
            "eval_error": eval_error,
            "fallback_error": fallback_error,
        }
        summary_rows.append(row)

    write_csv(
        tables / "metrics_summary.csv",
        summary_rows,
        [
            "model",
            "method",
            "train_seconds",
            "config_path",
            "psnr",
            "ssim",
            "lpips",
            "custom_psnr",
            "custom_ssim",
            "metric_source",
            "metric_log_tags",
            "eval_error",
            "fallback_error",
        ],
    )
    if detail_rows:
        write_csv(tables / "metrics_detail.csv", detail_rows, ["model", "metric", "value"])
    else:
        with (tables / "metrics_detail.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "metric", "value"])

    print(f"Metrics summary: {tables / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
