#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config_colmap_all.json"
SKIP_TRAIN=0
SKIP_EVAL=0
SKIP_COLMAP=0
CONVERT_EXISTING_COLMAP=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./run_colmap_all_pipeline.sh [options]

Options:
  --skip-train       Stop after COLMAP and split generation
  --skip-eval        Skip evaluation/report assets
  --skip-colmap      Reuse existing data/toy_ns_colmap_all/transforms.json
  --convert-existing-colmap
                     Convert the largest existing COLMAP sparse model, then continue
  --dry-run          Print external COLMAP/Nerfstudio commands without executing them
  -h, --help         Show this help

This pipeline sends every extracted raw frame to COLMAP:
- no frame filtering
- no crop
- no mask
- COLMAP camera model: SIMPLE_RADIAL
- COLMAP single camera mode: enabled
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --skip-colmap)
      SKIP_COLMAP=1
      shift
      ;;
    --convert-existing-colmap)
      CONVERT_EXISTING_COLMAP=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

run_step() {
  local title="$1"
  shift
  echo
  echo "==== ${title} ===="
  python "$@"
}

VIDEO_PATH="${SCRIPT_DIR}/data/raw/toy.mp4"
if [[ ! -f "$VIDEO_PATH" ]]; then
  mkdir -p "$(dirname "$VIDEO_PATH")"
  echo "Missing video file: $VIDEO_PATH" >&2
  exit 1
fi

if [[ "$CONVERT_EXISTING_COLMAP" -eq 1 ]]; then
  COLMAP_ARGS=(scripts/02_process_colmap_direct.py --config "$CONFIG" --convert-existing)
  if [[ "$DRY_RUN" -eq 1 ]]; then
    COLMAP_ARGS+=(--dry-run)
  fi
  run_step "1-2. Convert largest existing COLMAP sparse model" "${COLMAP_ARGS[@]}"
elif [[ "$SKIP_COLMAP" -eq 0 ]]; then
  run_step "1. Extract all raw frames" scripts/00_extract_score_frames.py --config "$CONFIG"

  COLMAP_ARGS=(scripts/02_process_colmap_direct.py --config "$CONFIG")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    COLMAP_ARGS+=(--dry-run)
  fi
  run_step "2. Direct COLMAP with SIMPLE_RADIAL + single_camera" "${COLMAP_ARGS[@]}"
else
  echo
  echo "==== 1-2. Reuse existing COLMAP output ===="
  if [[ ! -f "${SCRIPT_DIR}/data/toy_ns_colmap_all/transforms.json" ]]; then
    echo "Missing data/toy_ns_colmap_all/transforms.json; run without --skip-colmap first." >&2
    exit 1
  fi
fi

run_step "3. Pose-aware split from COLMAP poses" scripts/03_pose_stratified_split.py --config "$CONFIG"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  TRAIN_ARGS=(scripts/04_train_all.py --config "$CONFIG")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    TRAIN_ARGS+=(--dry-run)
  fi
  run_step "4. Train NeRF and 3DGS models" "${TRAIN_ARGS[@]}"
else
  echo "Skipping model training."
fi

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  EVAL_ARGS=(scripts/05_eval_metrics.py --config "$CONFIG" --skip-ns-eval)
  if [[ "$DRY_RUN" -eq 1 ]]; then
    EVAL_ARGS+=(--dry-run)
  fi
  run_step "5. Evaluate trained models" "${EVAL_ARGS[@]}"
  run_step "6. Build figures" scripts/06_make_figures.py --config "$CONFIG"
  run_step "7. Build qualitative examples" scripts/08_make_qualitative_examples.py --config "$CONFIG"
  run_step "8. Write report outline" scripts/07_write_report_outline.py --config "$CONFIG"
else
  echo "Skipping evaluation/report assets."
fi

echo
echo "COLMAP-all pipeline finished. Check:"
echo "  data/toy_ns_colmap_all/colmap_direct_summary.json"
echo "  data/toy_ns_colmap_all/transforms.json"
echo "  results_colmap_all/"
