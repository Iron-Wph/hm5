#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config.json"
INSTALL=0
SKIP_COLMAP=0
SKIP_TRAIN=0
SKIP_EVAL=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./run_pipeline.sh [options]

Options:
  --config PATH      Config file path. Default: config.json
  --install          Install Python helper packages from requirements.txt
  --skip-colmap      Skip Nerfstudio/COLMAP data processing
  --skip-train       Skip model training
  --skip-eval        Skip model evaluation
  --dry-run          Print external Nerfstudio commands without executing them
  -h, --help         Show this help

Examples:
  ./run_pipeline.sh
  ./run_pipeline.sh --skip-train --skip-eval
  ./run_pipeline.sh --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:?Missing value for --config}"
      shift 2
      ;;
    --install)
      INSTALL=1
      shift
      ;;
    --skip-colmap)
      SKIP_COLMAP=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
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

if [[ "$INSTALL" -eq 1 ]]; then
  echo "Installing Python helper packages from requirements.txt"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo
  echo "Nerfstudio/CUDA packages are intentionally not installed here because they depend on your GPU/CUDA setup."
  echo "Install them using the commands in 一键完成式实验执行说明.md before running training."
fi

VIDEO_PATH="${SCRIPT_DIR}/data/raw/toy.mp4"
if [[ ! -f "$VIDEO_PATH" ]]; then
  mkdir -p "$(dirname "$VIDEO_PATH")"
  echo "Missing video file: $VIDEO_PATH" >&2
  echo "Put toy.mp4 at this path, then run this script again." >&2
  exit 1
fi

run_step "1. Extract and score frames" scripts/00_extract_score_frames.py --config "$CONFIG"
run_step "2. Filter frames" scripts/01_filter_frames.py --config "$CONFIG"

if [[ "$SKIP_COLMAP" -eq 0 ]]; then
  COLMAP_ARGS=(scripts/02_process_colmap.py --config "$CONFIG")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    COLMAP_ARGS+=(--dry-run)
  fi
  run_step "3. COLMAP/Nerfstudio data processing" "${COLMAP_ARGS[@]}"
else
  echo "Skipping COLMAP/Nerfstudio data processing."
fi

run_step "4. Pose-aware split" scripts/03_pose_stratified_split.py --config "$CONFIG"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  TRAIN_ARGS=(scripts/04_train_all.py --config "$CONFIG")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    TRAIN_ARGS+=(--dry-run)
  fi
  run_step "5. Train NeRF and 3DGS models" "${TRAIN_ARGS[@]}"
else
  echo "Skipping model training."
fi

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  EVAL_ARGS=(scripts/05_eval_metrics.py --config "$CONFIG")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    EVAL_ARGS+=(--dry-run)
  fi
  run_step "6. Evaluate trained models" "${EVAL_ARGS[@]}"
else
  echo "Skipping evaluation."
fi

run_step "7. Build figures" scripts/06_make_figures.py --config "$CONFIG"
run_step "8. Build qualitative examples" scripts/08_make_qualitative_examples.py --config "$CONFIG"
run_step "9. Write report outline" scripts/07_write_report_outline.py --config "$CONFIG"

echo
echo "Pipeline finished. Check:"
echo "  results/tables/"
echo "  results/figures/"
echo "  report/final_report_outline.md"
