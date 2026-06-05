#!/usr/bin/env bash
set -euo pipefail

# Calculate PSNR, SSIM, and LPIPS against baseline Open-Sora videos.
# Usage:
#   cd /path/to/NaviCache
#   bash scripts/opensora/run_opensora_common_metrics.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVICACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${NAVICACHE_ROOT}"

: "${GT_VIDEO_DIR:=navicache4opensora/eval/navicache/samples/opensora_base}"
: "${GENERATED_VIDEO_DIR:=navicache4opensora/eval/navicache/samples/opensora_navicache}"

python navicache4opensora/eval/navicache/common_metrics/eval.py \
    --gt_video_dir "${GT_VIDEO_DIR}" \
    --generated_video_dir "${GENERATED_VIDEO_DIR}"
