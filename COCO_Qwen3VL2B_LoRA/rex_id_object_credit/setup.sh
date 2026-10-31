#!/usr/bin/env bash
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
bash "$ROOT/rex_id_grpo/setup.sh"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
PIP_CONFIG_FILE=/dev/null python -m pip install --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" scipy==1.13.1
