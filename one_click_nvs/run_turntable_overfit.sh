#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config_turntable_overfit.json"
SKIP_TRAIN=0
SKIP_EVAL=0
FORCE_RETRAIN=0

usage() {
  cat <<'EOF'
Usage:
  ./run_turntable_overfit.sh [options]

Options:
  --skip-train       Generate full-video turntable dataset but skip training
  --skip-eval        Skip evaluation/report assets
  --force-retrain    Retrain even if configs already exist
  -h, --help         Show this help

This pipeline is for full-video long-training quality probing:
- frame_step=1
- all filtered frames are used for training
- 30000 iterations

Do not use its metrics as strict held-out evaluation because train_all_frames=true.
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

VIDEO_PATH="${SCRIPT_DIR}/data/raw/toy.mp4"
if [[ ! -f "$VIDEO_PATH" ]]; then
  mkdir -p "$(dirname "$VIDEO_PATH")"
  echo "Missing video file: $VIDEO_PATH" >&2
  exit 1
fi

run_step "1. Extract all frames" scripts/00_extract_score_frames.py --config "$CONFIG"
run_step "2. Light frame filtering" scripts/01_filter_frames.py --config "$CONFIG"
run_step "3. Generate object-focused full-video turntable dataset" scripts/02b_generate_turntable_poses.py --config "$CONFIG"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  TRAIN_ARGS=(scripts/04_train_all.py --config "$CONFIG")
  if [[ "$FORCE_RETRAIN" -eq 1 ]]; then
    TRAIN_ARGS+=(--force-retrain)
  fi
  run_step "4. Train full-video models" "${TRAIN_ARGS[@]}"
else
  echo "Skipping model training."
fi

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  run_step "5. Evaluate full-video models" scripts/05_eval_metrics.py --config "$CONFIG" --skip-ns-eval
  run_step "6. Build figures" scripts/06_make_figures.py --config "$CONFIG"
  run_step "7. Build qualitative examples" scripts/08_make_qualitative_examples.py --config "$CONFIG"
  run_step "8. Write report outline" scripts/07_write_report_outline.py --config "$CONFIG"
else
  echo "Skipping evaluation/report assets."
fi

echo
echo "Full-video pipeline finished. Check:"
echo "  results_turntable_overfit/figures/qualitative_examples.png"
echo "  results_turntable_overfit/tables/metrics_summary.csv"
