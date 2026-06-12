from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from common import ensure_dir, load_config, relpath


def markdown_table(path: Path) -> str:
    if not path.exists():
        return "_待生成_"
    df = pd.read_csv(path)
    if df.empty:
        return "_暂无数据_"
    df = df.fillna("")
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.astype(str).values.tolist()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_line(headers), separator, *(fmt_line(row) for row in rows)])


def rel_link(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target, start=from_dir)).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report_dir = ensure_dir(relpath(cfg, "paths", "report"))
    tables_dir = relpath(cfg, "paths", "tables")
    figures_dir = relpath(cfg, "paths", "figures")
    splits_dir = relpath(cfg, "paths", "splits")

    metrics_md = markdown_table(tables_dir / "metrics_summary.csv")
    split_md = markdown_table(splits_dir / "split.csv")
    frame_quality_fig = rel_link(report_dir, figures_dir / "frame_quality_distribution.png")
    trajectory_fig = rel_link(report_dir, figures_dir / "camera_trajectory_split.png")
    psnr_fig = rel_link(report_dir, figures_dir / "psnr_barplot.png")
    ssim_fig = rel_link(report_dir, figures_dir / "ssim_barplot.png")
    lpips_fig = rel_link(report_dir, figures_dir / "lpips_barplot.png")
    speed_quality_fig = rel_link(report_dir, figures_dir / "speed_quality_tradeoff.png")
    is_turntable = "turntable" in cfg.get("project_name", "").lower() or "turntable" in str(relpath(cfg, "paths", "nerfstudio_data")).lower()
    pose_method = "转台/圆轨迹先验位姿" if is_turntable else "COLMAP/Nerfstudio"
    run_command = "./run_turntable_pipeline.sh" if is_turntable else "./run_pipeline.sh"

    text = f"""# toy.mp4 新视图合成实验报告草稿

## 1. Introducao 引言

本实验基于 `toy.mp4` 视频完成新视图合成任务。实验比较 NeRF 类方法 `nerfacto` 与 3D Gaussian Splatting 类方法 `splatfacto`，并从重建质量、视觉真实感、训练成本、渲染效率和伪影类型等角度进行综合分析。

本实验的创新性体现在：

- 自动帧质量筛选：根据模糊度、亮度、对比度和重复度筛选视频帧。
- 轨迹感知数据划分：按照相机轨迹进行 `60:20:20` 训练、验证、测试划分。
- 多维评价：除 PSNR、SSIM、LPIPS 外，还预留主体区域评价和误差热力图分析。
- 速度-质量权衡：将训练时间与质量指标共同纳入模型比较。

## 2. Dataset e Tecnologia 数据集与技术

数据来源：`toy.mp4`

处理流程：

1. 从视频中按固定间隔抽帧。
2. 对每一帧计算质量指标。
3. 删除模糊、曝光异常和重复帧。
4. 使用 {pose_method} 生成相机位姿。
5. 生成 NeRF/3DGS 可训练的数据格式。

帧质量分布图：

![frame_quality_distribution]({frame_quality_fig})

相机轨迹与划分图：

![camera_trajectory_split]({trajectory_fig})

## 3. Metodologia 方法论

### 3.1 自动帧筛选

本实验使用 Laplacian 方差衡量图像模糊度，使用灰度均值衡量亮度，使用灰度标准差衡量对比度，并通过感知哈希检测相邻帧重复度。

### 3.2 轨迹感知划分

视频相邻帧高度相似，随机划分容易导致测试指标虚高。因此本实验根据相机轨迹顺序进行分层采样，形成训练集、验证集和测试集。

本次配置采用的位姿策略：{pose_method}。

### 3.3 模型设置

NeRF 类模型：

- `nerfacto`
- `nerfacto-big`

3DGS 类模型：

- `splatfacto`
- `splatfacto + scale regularization`

### 3.4 评价指标

- PSNR：衡量像素级重建误差，越高越好。
- SSIM：衡量结构相似性，越高越好。
- LPIPS：衡量感知相似度，越低越好。
- 训练时间：衡量计算成本。
- 误差热力图：定位模型失败区域。

## 4. Resultados 结果

### 4.1 数据划分

{split_md}

### 4.2 模型指标

{metrics_md}

### 4.3 指标图

![psnr_barplot]({psnr_fig})

![ssim_barplot]({ssim_fig})

![lpips_barplot]({lpips_fig})

![speed_quality_tradeoff]({speed_quality_fig})

### 4.4 结果讨论

请根据实际渲染图像补充以下分析：

- 哪个模型在 PSNR/SSIM/LPIPS 上表现最好。
- 哪个模型的新视图更自然。
- NeRF 是否出现模糊或细节不足。
- 3DGS 是否出现漂浮高斯、尖刺、边缘伪影。
- 清洗帧策略是否改善 COLMAP 定位和最终重建质量。
- 指标结果与人眼观察是否一致。

## 5. Conclusoes 总结

请根据最终结果给出明确结论：

- 如果 3DGS 速度显著更快且质量接近 NeRF，可推荐 `splatfacto`。
- 如果 NeRF 细节和稳定性更好，可推荐 `nerfacto` 或 `nerfacto-big`。
- 如果某个创新策略提升明显，应强调自动帧筛选或轨迹感知划分对结果可信度的贡献。

## 附录：复现实验命令

```bash
cd one_click_nvs
{run_command}
```
"""

    out = report_dir / "final_report_outline.md"
    out.write_text(text, encoding="utf-8")
    print(f"Report outline: {out}")


if __name__ == "__main__":
    main()
