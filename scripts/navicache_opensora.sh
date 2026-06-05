#!/usr/bin/env bash
set -euo pipefail

# Run the default Open-Sora NaviCache evaluation pipeline.
# Usage:
#   cd /path/to/NaviCache
#   bash scripts/navicache_opensora.sh
#
# This launcher runs generation, VBench scoring, and common metrics in order.
# To run a single stage, call a script under scripts/opensora/ directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/opensora/run_opensora_generate.sh"
bash "${SCRIPT_DIR}/opensora/run_opensora_vbench.sh"
bash "${SCRIPT_DIR}/opensora/run_opensora_common_metrics.sh"
