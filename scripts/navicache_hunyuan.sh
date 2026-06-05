#!/usr/bin/env bash
set -euo pipefail

# Run the default HunyuanVideo NaviCache examples.
# Usage:
#   cd /path/to/HunyuanVideo
#   bash /path/to/NaviCache/scripts/navicache_hunyuan.sh
#
# To run a single example, call a script under scripts/hunyuan/ directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/hunyuan/run_hunyuan_544p.sh"
bash "${SCRIPT_DIR}/hunyuan/run_hunyuan_720p.sh"
