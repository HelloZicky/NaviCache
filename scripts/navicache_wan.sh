#!/usr/bin/env bash
set -euo pipefail

# Run the default Wan2.1 NaviCache examples.
# Usage:
#   cd /path/to/Wan2.1
#   bash /path/to/NaviCache/scripts/navicache_wan.sh
#
# To run a single example, call a script under scripts/wan/ directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/wan/run_wan_t2v_1.3b.sh"
bash "${SCRIPT_DIR}/wan/run_wan_i2v_480p.sh"
