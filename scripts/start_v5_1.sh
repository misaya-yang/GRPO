#!/usr/bin/env bash
set -euo pipefail
V5_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V5_ROOT"
export PYTHONPATH="$V5_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
V5_PYTHON="${V5_PYTHON:-python}"
exec "$V5_PYTHON" -u scripts/v5_1_control.py "${@:-status}"
