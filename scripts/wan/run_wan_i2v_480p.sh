#!/usr/bin/env bash
set -euo pipefail

# Run NaviCache on Wan2.1 image-to-video with the 480P checkpoint.
# Usage:
#   cd /path/to/Wan2.1
#   bash /path/to/NaviCache/scripts/wan/run_wan_i2v_480p.sh
#
# Common overrides:
#   IMAGE=/path/to/input.jpg CKPT_DIR=/path/to/Wan2.1-I2V-14B-480P bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVICACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Copy the NaviCache Wan2.1 entry script into the official Wan2.1 repo.
cp "${NAVICACHE_ROOT}/NaviCache4Wan2.1/navicache_generate.py" ./navicache_generate.py

: "${TASK:=i2v-14B}"
: "${SIZE:=832*480}"
: "${CKPT_DIR:=./Wan2.1-I2V-14B-480P}"
: "${IMAGE:=examples/i2v_input.JPG}"
: "${NAVICACHE_THRESH:=0.05}"
: "${NAVICACHE_ALIGN_STEPS:=10}"
: "${PROMPT:=Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds.}"

python navicache_generate.py \
    --task "${TASK}" \
    --size "${SIZE}" \
    --ckpt_dir "${CKPT_DIR}" \
    --image "${IMAGE}" \
    --offload_model True \
    --t5_cpu \
    --navicache_thresh "${NAVICACHE_THRESH}" \
    --navicache_align_steps "${NAVICACHE_ALIGN_STEPS}" \
    --prompt "${PROMPT}"
