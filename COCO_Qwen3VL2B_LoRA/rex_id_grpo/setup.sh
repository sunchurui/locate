#!/usr/bin/env bash
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
PIP_CONFIG_FILE=/dev/null python -m pip install --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
  ms-swift==4.6.0 transformers==4.57.6 peft==0.20.0 trl==0.29.1 msgspec==0.22.0 matplotlib
REX="$ROOT/references/Rex-Omni"
REV=6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e
if [[ ! -d "$REX/.git" ]]; then
  mkdir -p "$ROOT/references"
  git clone --filter=blob:none --no-checkout https://github.com/IDEA-Research/Rex-Omni.git "$REX"
  git -C "$REX" sparse-checkout set finetuning
  git -C "$REX" checkout "$REV"
fi
if [[ "$(git -C "$REX" rev-parse HEAD)" != "$REV" ]]; then
  printf 'Rex source differs from pinned revision %s; keep a separate checkout.\n' "$REV" >&2
  exit 1
fi
test -f "$REX/finetuning/verl/configs/reward_func.py"
