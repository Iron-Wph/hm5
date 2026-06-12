#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config_turntable.json"
INSTALL=0
SKIP_TRAIN=0
SKIP_EVAL=0
FORCE_RETRAIN=0

usage() {
  cat <<'EOF'
Usage:
  ./run_turntable_pipeline.sh [options]

Options:
  --config PATH      Config file path. Default: config_turntable.json
  --install          Install Python helper packages from requirements.txt
  --skip-train       Skip model training
  --skip-eval        Skip model evaluation
  --force-retrain    Retrain even if model configs already exist
  -h, --help         Show this help
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
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --force-retrain)
      FORCE_RETRAIN=1
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
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

VIDEO_PATH="${SCRIPT_DIR}/data/raw/toy.mp4"
if [[ ! -f "$VIDEO_PATH" ]]; then
  mkdir -p "$(dirname "$VIDEO_PATH")"
  echo "Missing video file: $VIDEO_PATH" >&2
  exit 1
fi

if [[ ! -d data/frames_clean ]] || [[ -z "$(find data/frames_clean -maxdepth 1 -type f 2>/dev/null | head -n 1)" ]]; then
  run_step "1. Extract and score frames" scripts/00_extract_score_frames.py --config "$CONFIG"
  run_step "2. Filter frames" scripts/01_filter_frames.py --config "$CONFIG"
else
  echo "Using existing data/frames_clean. Remove it if you want to re-extract frames."
fi

run_step "3. Generate turntable pose-prior dataset" scripts/02b_generate_turntable_poses.py --config "$CONFIG"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  TRAIN_ARGS=(scripts/04_train_all.py --config "$CONFIG")
  if [[ "$FORCE_RETRAIN" -eq 1 ]]; then
    TRAIN_ARGS+=(--force-retrain)
  fi
  run_step "4. Train NeRF and 3DGS with turntable poses" "${TRAIN_ARGS[@]}"
else
  echo "Skipping model training."
fi

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  run_step "5. Evaluate models" scripts/05_eval_metrics.py --config "$CONFIG" --skip-ns-eval
else
  echo "Skipping evaluation."
fi

run_step "6. Build figures" scripts/06_make_figures.py --config "$CONFIG"
run_step "7. Write report outline" scripts/07_write_report_outline.py --config "$CONFIG"

echo
echo "Turntable pipeline finished. Check:"
echo "  results_turntable/tables/"
echo "  results_turntable/figures/"
echo "  report_turntable/final_report_outline.md"
