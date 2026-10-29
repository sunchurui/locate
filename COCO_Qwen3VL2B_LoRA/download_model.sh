#!/usr/bin/env bash
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
if [[ -s "$MODEL_DIR/model.safetensors" && -f "$MODEL_DIR/config.json" ]]; then
  echo "Qwen3-VL-2B-Instruct already exists at $MODEL_DIR"
  exit 0
fi
mkdir -p "$MODEL_DIR"
"$ROOT/venv/bin/modelscope" download --model Qwen/Qwen3-VL-2B-Instruct --local_dir "$MODEL_DIR"
test -s "$MODEL_DIR/model.safetensors"
test -f "$MODEL_DIR/config.json"
echo "Model files verified in $MODEL_DIR"
