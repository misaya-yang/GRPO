#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
research_python="${RESEARCH_PYTHON:-/root/miniconda3/bin/python}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "$research_python" -m dependent_rollouts.cli "$@"
