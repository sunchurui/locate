#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=coco_qwen2b_base
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
DATA="$ROOT/data/coco2017"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
OUT="$ROOT/evaluation/base_val2017_full"
BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

source "${VENV_PATH:-$ROOT/venv}/bin/activate"
mkdir -p "$ROOT/logs" "$OUT"
cd "$ROOT"
export QWENVL_BBOX_FORMAT=new
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python "$ROOT/evaluate_coco.py" \
  --model "$MODEL_DIR" \
  --annotations "$DATA/annotations/instances_val2017.json" \
  --image-dir "$DATA/val2017" \
  --output-dir "$OUT" \
  --batch-size "$BATCH_SIZE"

echo "Full COCO validation for the unmodified base model completed."
