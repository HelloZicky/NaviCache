#!/usr/bin/env bash
set -euo pipefail

# Run NaviCache on HunyuanVideo at 544 x 960 resolution.
# Usage:
#   cd /path/to/HunyuanVideo
#   bash /path/to/NaviCache/scripts/hunyuan/run_hunyuan_544p.sh
#
# Common overrides:
#   PROMPT="A cinematic shot of a corgi running through a snowy forest." bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVICACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Copy the NaviCache HunyuanVideo entry script into the official HunyuanVideo repo.
cp "${NAVICACHE_ROOT}/NaviCache4HunyuanVideo/navicache_sample_video.py" ./navicache_sample_video.py

: "${VIDEO_HEIGHT:=544}"
: "${VIDEO_WIDTH:=960}"
: "${VIDEO_LENGTH:=129}"
: "${INFER_STEPS:=50}"
: "${NAVICACHE_THRESH:=0.025}"
: "${NAVICACHE_ALIGN_STEPS:=5}"
: "${SAVE_PATH:=./navicache_results}"
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
