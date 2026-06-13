from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from common import ensure_dir, load_config, relpath


BAD_TOKENS = ["depth", "accumulation", "weights", "mask", "normal"]
GOOD_TOKENS = ["img", "rgb", "render", "prediction"]


def score_image(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    score = 0
    if any(token in name for token in GOOD_TOKENS):
        score += 20
    if any(token in name for token in BAD_TOKENS):
        score -= 100
    if "gt" in name or "ground" in name:
        score += 5
    return score, path.name


def sample_id(path: Path) -> str:
    match = re.search(r"_(\d{4,6})_", path.stem)
    if match:
        return match.group(1)
    match = re.search(r"(\d{4,6})", path.stem)
    if match:
        return match.group(1)
    return "0000"


def model_names_from_summary(tables_dir: Path, results_dir: Path) -> list[str]:
    metrics = tables_dir / "metrics_summary.csv"
    if metrics.exists():
        df = pd.read_csv(metrics)
        if "model" in df.columns:
            return [str(name) for name in df["model"].dropna().tolist()]
    return sorted([p.name for p in results_dir.iterdir() if (p / "renders").exists()])


def collect_model_images(results_dir: Path, model: str) -> dict[str, Path]:
    render_dir = results_dir / model / "renders"
    if not render_dir.exists():
        return {}
    images = [p for p in render_dir.rglob("*.png") if score_image(p)[0] > -50]
    grouped: dict[str, list[Path]] = {}
    for path in images:
        grouped.setdefault(sample_id(path), []).append(path)
    selected: dict[str, Path] = {}
    for sid, paths in grouped.items():
        selected[sid] = max(paths, key=score_image)
    return selected


def load_cell(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def make_panel(model_to_images: dict[str, dict[str, Path]], out_path: Path, max_samples: int = 3) -> bool:
    models = [model for model, images in model_to_images.items() if images]
    if not models:
        return False

    common_ids = set(model_to_images[models[0]].keys())
    for model in models[1:]:
        common_ids &= set(model_to_images[model].keys())
    sample_ids = sorted(common_ids)[:max_samples]
    if not sample_ids:
        sample_ids = sorted({sid for images in model_to_images.values() for sid in images})[:max_samples]
    if not sample_ids:
        return False

    cell = (420, 260)
    label_h = 44
    left_w = 110
    width = left_w + len(models) * cell[0]
    height = label_h + len(sample_ids) * (cell[1] + label_h)
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)

    for col, model in enumerate(models):
        draw_label(draw, (left_w + col * cell[0] + 12, 12), model)

    for row, sid in enumerate(sample_ids):
        y0 = label_h + row * (cell[1] + label_h)
        draw_label(draw, (12, y0 + 12), f"sample {sid}")
        for col, model in enumerate(models):
            path = model_to_images[model].get(sid)
            x0 = left_w + col * cell[0]
            draw.rectangle([x0, y0, x0 + cell[0] - 1, y0 + cell[1] - 1], outline=(220, 220, 220))
            if path:
                panel.paste(load_cell(path, cell), (x0, y0))
                draw_label(draw, (x0 + 8, y0 + cell[1] + 8), path.name[:44])
            else:
                draw_label(draw, (x0 + 20, y0 + 110), "missing")

    ensure_dir(out_path.parent)
    panel.save(out_path)
    return True


def copy_examples(model_to_images: dict[str, dict[str, Path]], out_dir: Path, max_per_model: int = 4) -> None:
    ensure_dir(out_dir)
    for model, images in model_to_images.items():
        model_dir = ensure_dir(out_dir / model)
        for sid, path in sorted(images.items())[:max_per_model]:
            shutil.copy2(path, model_dir / f"sample_{sid}_{path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_turntable.json")
    parser.add_argument("--max-samples", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = relpath(cfg, "paths", "results")
    figures_dir = ensure_dir(relpath(cfg, "paths", "figures"))
    tables_dir = relpath(cfg, "paths", "tables")

    models = model_names_from_summary(tables_dir, results_dir)
    model_to_images = {model: collect_model_images(results_dir, model) for model in models}
    copy_examples(model_to_images, figures_dir / "qualitative_examples")
    out_path = figures_dir / "qualitative_examples.png"
    if make_panel(model_to_images, out_path, max_samples=args.max_samples):
        print(f"Qualitative panel written to {out_path}")
    else:
        print("No render images found for qualitative panel.")
        print(f"Look under {results_dir}/<model>/renders after evaluation.")


if __name__ == "__main__":
    main()
