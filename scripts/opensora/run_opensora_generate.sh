#!/usr/bin/env bash
set -euo pipefail

# Generate Open-Sora baseline and NaviCache videos for evaluation.
# Usage:
#   cd /path/to/NaviCache
#   bash scripts/opensora/run_opensora_generate.sh
#
# Common overrides:
#   NAVICACHE_THRESH=0.55 NAVICACHE_ALIGN_STEPS=5 PROMPT_PATH=/path/to/prompts.json bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVICACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${NAVICACHE_ROOT}"

: "${TRANSFORMER_PATH:=checkpoints/opensora/OpenSora-STDiT-v3}"
: "${VAE_PATH:=checkpoints/opensora/OpenSora-VAE-v1.2}"
: "${TEXT_ENCODER_PATH:=checkpoints/opensora/t5-v1_1-xxl}"
: "${PROMPT_PATH:=eval/navicache/vbench/VBench_full_info.json}"
: "${NAVICACHE_THRESH:=0.35}"
: "${NAVICACHE_ALIGN_STEPS:=5}"

python eval/navicache/experiments/opensora.py \
    --transformer_path "${TRANSFORMER_PATH}" \
    --vae_path "${VAE_PATH}" \
    --text_encoder_path "${TEXT_ENCODER_PATH}" \
    --prompt_path "${PROMPT_PATH}" \
    --navicache_thresh "${NAVICACHE_THRESH}" \
    --navicache_align_steps "${NAVICACHE_ALIGN_STEPS}"
