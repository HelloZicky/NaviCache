#!/usr/bin/env bash
set -euo pipefail

# Run NaviCache on Wan2.1 text-to-video with the 1.3B checkpoint.
# Usage:
#   cd /path/to/Wan2.1
#   bash /path/to/NaviCache/scripts/wan/run_wan_t2v_1.3b.sh
#
# Common overrides:
#   CKPT_DIR=/path/to/Wan2.1-T2V-1.3B NAVICACHE_THRESH=0.07 bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVICACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Copy the NaviCache Wan2.1 entry script into the official Wan2.1 repo.
cp "${NAVICACHE_ROOT}/NaviCache4Wan2.1/navicache_generate.py" ./navicache_generate.py

: "${TASK:=t2v-1.3B}"
: "${SIZE:=832*480}"
: "${CKPT_DIR:=./Wan2.1-T2V-1.3B}"
: "${NAVICACHE_THRESH:=0.05}"
: "${NAVICACHE_ALIGN_STEPS:=10}"
: "${PROMPT:=Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"

python navicache_generate.py \
    --task "${TASK}" \
    --size "${SIZE}" \
    --ckpt_dir "${CKPT_DIR}" \
    --offload_model True \
    --t5_cpu \
    --navicache_thresh "${NAVICACHE_THRESH}" \
    --navicache_align_steps "${NAVICACHE_ALIGN_STEPS}" \
    --prompt "${PROMPT}"
