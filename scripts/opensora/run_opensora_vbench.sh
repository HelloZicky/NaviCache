#!/usr/bin/env bash
set -euo pipefail

# Calculate VBench scores for generated NaviCache Open-Sora videos.
# Usage:
#   cd /path/to/NaviCache
#   bash scripts/opensora/run_opensora_vbench.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVICACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${NAVICACHE_ROOT}"

: "${VIDEO_PATH:=NaviCache4OpenSora/eval/navicache/samples/opensora_navicache}"
: "${SAVE_PATH:=NaviCache4OpenSora/eval/navicache/vbench_results/navicache}"
: "${SCORE_DIR:=${SAVE_PATH}}"

python NaviCache4OpenSora/eval/navicache/vbench/run_vbench.py \
    --video_path "${VIDEO_PATH}" \
    --save_path "${SAVE_PATH}"

python NaviCache4OpenSora/eval/navicache/vbench/cal_vbench.py \
    --score_dir "${SCORE_DIR}"
