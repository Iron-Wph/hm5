from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

from common import ROOT, ensure_dir, load_config, relpath, require_command, run_command, write_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def is_within_root(path: Path) -> bool:
    resolved = path.resolve()
    root = ROOT.resolve()
    return resolved == root or root in resolved.parents


def safe_remove(path: Path) -> None:
    if not path.exists():
        return
    if not is_within_root(path):
        raise RuntimeError(f"Refusing to remove path outside project root: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_all_images(source_dir: Path, target_dir: Path) -> list[Path]:
    ensure_dir(target_dir)
    images = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise SystemExit(f"No images found in {source_dir}. Run 00_extract_score_frames.py first.")
    copied: list[Path] = []
    for src in images:
        dst = target_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def count_colmap_images(model_dir: Path) -> int:
    images_bin = model_dir / "images.bin"
    if not images_bin.exists():
        return 0
    with images_bin.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        return 0
    return int(struct.unpack("<Q", raw)[0])


def find_colmap_models(sparse_dir: Path) -> list[dict[str, object]]:
    if not sparse_dir.exists():
        return []
    models: list[dict[str, object]] = []
    for model_dir in sorted([p for p in sparse_dir.iterdir() if p.is_dir()]):
        registered_images = count_colmap_images(model_dir)
        if registered_images > 0:
            models.append({"model_dir": model_dir, "registered_images": registered_images})
    return models


def bool_flag(value: object, default: bool = False) -> str:
    if value is None:
        value = default
    return "1" if bool(value) else "0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_colmap_all.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--convert-existing",
        action="store_true",
        help="Skip COLMAP commands and convert the largest existing sparse model to transforms.json.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    colmap_cfg = cfg.get("colmap", {})
    source_key = str(colmap_cfg.get("source", "frames_raw"))
    source_dir = relpath(cfg, "paths", source_key)
    ns_data = ensure_dir(relpath(cfg, "paths", "nerfstudio_data"))
    image_dir = ns_data / "images"
    colmap_dir = ns_data / "colmap"
    sparse_dir = colmap_dir / "sparse"
    database_path = colmap_dir / "database.db"

    overwrite = bool(colmap_cfg.get("overwrite", False))
    if overwrite and not args.dry_run and not args.convert_existing:
        safe_remove(image_dir)
        safe_remove(colmap_dir)
        safe_remove(ns_data / "transforms.json")
        safe_remove(ns_data / "sparse_pc.ply")

    if args.convert_existing:
        copied = list_images(image_dir)
        if not copied:
            copied = list_images(source_dir)
    else:
        copied = copy_all_images(source_dir, image_dir)
    ensure_dir(colmap_dir)

    if not args.dry_run and not args.convert_existing:
        require_command("colmap")

    camera_model = str(colmap_cfg.get("camera_model", "SIMPLE_RADIAL"))
    matching_method = str(colmap_cfg.get("matching_method", "sequential"))
    if matching_method not in {"sequential", "exhaustive", "vocab_tree"}:
        raise SystemExit(f"Unsupported matching_method: {matching_method}")

    use_gpu = bool_flag(colmap_cfg.get("use_gpu", True), True)
    single_camera = bool_flag(colmap_cfg.get("single_camera", True), True)
    guided_matching = bool_flag(colmap_cfg.get("guided_matching", True), True)

    if not args.convert_existing:
        feature_cmd = [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--ImageReader.single_camera",
            single_camera,
            "--ImageReader.camera_model",
            camera_model,
            "--SiftExtraction.use_gpu",
            use_gpu,
        ]
        sift_max_image_size = int(colmap_cfg.get("sift_max_image_size", 0))
        if sift_max_image_size > 0:
            feature_cmd.extend(["--SiftExtraction.max_image_size", str(sift_max_image_size)])
        run_command(feature_cmd, dry_run=args.dry_run)

        matcher_cmd = [
            "colmap",
            f"{matching_method}_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            use_gpu,
            "--SiftMatching.guided_matching",
            guided_matching,
        ]
        if matching_method == "sequential":
            matcher_cmd.extend(["--SequentialMatching.overlap", str(int(colmap_cfg.get("sequential_overlap", 20)))])
        run_command(matcher_cmd, dry_run=args.dry_run)

        ensure_dir(sparse_dir)
        mapper_cmd = [
            "colmap",
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
            "--Mapper.ba_refine_focal_length",
            bool_flag(colmap_cfg.get("ba_refine_focal_length", True), True),
            "--Mapper.ba_refine_principal_point",
            bool_flag(colmap_cfg.get("ba_refine_principal_point", False), False),
            "--Mapper.ba_refine_extra_params",
            bool_flag(colmap_cfg.get("ba_refine_extra_params", True), True),
        ]
        run_command(mapper_cmd, dry_run=args.dry_run)

    if args.dry_run:
        print(f"Dry run complete. Expected COLMAP sparse root: {sparse_dir}")
        return

    colmap_models = find_colmap_models(sparse_dir)
    if not colmap_models:
        raise SystemExit(f"COLMAP did not create any sparse model under {sparse_dir}")
    selected = max(colmap_models, key=lambda item: int(item["registered_images"]))
    model_dir = Path(selected["model_dir"])
    if not (model_dir / "cameras.bin").exists():
        raise SystemExit(f"Selected COLMAP model is missing cameras.bin: {model_dir}")

    try:
        from nerfstudio.process_data import colmap_utils
    except Exception as exc:
        raise SystemExit(
            "COLMAP finished, but Nerfstudio is required to convert the model to transforms.json. "
            "Run this script inside the Nerfstudio environment."
        ) from exc

    registered = colmap_utils.colmap_to_json(
        recon_dir=model_dir,
        output_dir=ns_data,
        use_single_camera_mode=bool(colmap_cfg.get("single_camera", True)),
    )
    write_json(
        ns_data / "colmap_direct_summary.json",
        {
            "input_image_dir": str(source_dir),
            "copied_image_dir": str(image_dir),
            "input_images": len(copied),
            "registered_images": registered,
            "registration_ratio": registered / len(copied) if copied else 0.0,
            "camera_model": camera_model,
            "single_camera": bool(colmap_cfg.get("single_camera", True)),
            "matching_method": matching_method,
            "sequential_overlap": int(colmap_cfg.get("sequential_overlap", 20)),
            "model_dir": str(model_dir),
            "all_colmap_models": [
                {"model_dir": str(item["model_dir"]), "registered_images": int(item["registered_images"])}
                for item in colmap_models
            ],
            "transforms": str(ns_data / "transforms.json"),
        },
    )
    print(f"COLMAP input images: {len(copied)}")
    print("COLMAP sparse models:")
    for item in colmap_models:
        print(f"  {item['model_dir']}: {item['registered_images']} registered images")
    print(f"Selected COLMAP model: {model_dir}")
    print(f"COLMAP registered images: {registered}")
    print(f"Registration ratio: {registered / len(copied):.2%}")
    print(f"Nerfstudio transforms: {ns_data / 'transforms.json'}")
    print(f"COLMAP summary: {ns_data / 'colmap_direct_summary.json'}")


if __name__ == "__main__":
    main()
