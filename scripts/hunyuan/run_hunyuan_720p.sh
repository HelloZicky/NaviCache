#!/usr/bin/env bash
set -euo pipefail

# Run NaviCache on HunyuanVideo at 720 x 1280 resolution.
# Usage:
#   cd /path/to/HunyuanVideo
#   bash /path/to/NaviCache/scripts/hunyuan/run_hunyuan_720p.sh
#
# Common overrides:
#   SAVE_PATH=./navicache_results_720p NAVICACHE_THRESH=0.025 bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVICACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Copy the NaviCache HunyuanVideo entry script into the official HunyuanVideo repo.
cp "${NAVICACHE_ROOT}/NaviCache4HunyuanVideo/navicache_sample_video.py" ./navicache_sample_video.py

: "${VIDEO_HEIGHT:=720}"
: "${VIDEO_WIDTH:=1280}"
: "${VIDEO_LENGTH:=129}"
: "${INFER_STEPS:=50}"
: "${NAVICACHE_THRESH:=0.025}"
: "${NAVICACHE_ALIGN_STEPS:=5}"
: "${SAVE_PATH:=./navicache_results_720p}"
: "${PROMPT:=A cat walks on the grass, realistic style.}"

python3 navicache_sample_video.py \
    --video-size "${VIDEO_HEIGHT}" "${VIDEO_WIDTH}" \
    --video-length "${VIDEO_LENGTH}" \
    --infer-steps "${INFER_STEPS}" \
    --prompt "${PROMPT}" \
    --use-cpu-offload \
    --navicache_thresh "${NAVICACHE_THRESH}" \
    --navicache_align_steps "${NAVICACHE_ALIGN_STEPS}" \
    --save-path "${SAVE_PATH}"
