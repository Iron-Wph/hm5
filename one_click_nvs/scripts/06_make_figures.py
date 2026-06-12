from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ensure_dir, load_config, relpath


def save_frame_quality(splits_dir: Path, figures_dir: Path) -> None:
    csv_path = splits_dir / "frame_quality.csv"
    if not csv_path.exists():
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(df["blur"], ax=axes[0], kde=True)
    axes[0].set_title("Blur score")
    sns.histplot(df["brightness"], ax=axes[1], kde=True)
    axes[1].set_title("Brightness")
    sns.histplot(df["contrast"], ax=axes[2], kde=True)
    axes[2].set_title("Contrast")
    fig.tight_layout()
    fig.savefig(figures_dir / "frame_quality_distribution.png", dpi=180)
    plt.close(fig)


def save_trajectory(splits_dir: Path, figures_dir: Path) -> None:
    split_path = splits_dir / "split.csv"
    if not split_path.exists():
        return
    import matplotlib.pyplot as plt

    df = pd.read_csv(split_path)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"train": "#2E7D32", "val": "#1565C0", "test": "#C62828"}
    for split, group in df.groupby("split"):
        ax.scatter(group["x"], group["y"], group["z"], s=22, label=split, color=colors.get(split))
    ax.set_title("Camera trajectory split")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "camera_trajectory_split.png", dpi=180)
    plt.close(fig)


def save_metrics(tables_dir: Path, figures_dir: Path) -> None:
    metrics_path = tables_dir / "metrics_summary.csv"
    if not metrics_path.exists():
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(metrics_path)
    metric_cols = [c for c in ["psnr", "ssim", "lpips", "custom_psnr", "custom_ssim"] if c in df.columns]
    for col in metric_cols:
        if df[col].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(data=df, x="model", y=col, ax=ax)
        ax.set_title(col.upper())
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{col}_barplot.png", dpi=180)
        plt.close(fig)

    if "train_seconds" in df.columns and "psnr" in df.columns and df["psnr"].notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=df, x="train_seconds", y="psnr", hue="model", s=90, ax=ax)
        ax.set_title("Speed-quality tradeoff")
        ax.set_xlabel("Training seconds")
        ax.set_ylabel("PSNR")
        fig.tight_layout()
        fig.savefig(figures_dir / "speed_quality_tradeoff.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_dir = relpath(cfg, "paths", "splits")
    tables_dir = ensure_dir(relpath(cfg, "paths", "tables"))
    figures_dir = ensure_dir(relpath(cfg, "paths", "figures"))

    save_frame_quality(splits_dir, figures_dir)
    save_trajectory(splits_dir, figures_dir)
    save_metrics(tables_dir, figures_dir)
    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
