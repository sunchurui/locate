#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=coco_rex_id_eval
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA="$ROOT/data/coco2017"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
ADAPTER="${ADAPTER:-$ROOT/rex_id_sft_outputs}"
OUTPUT="${EVAL_OUTPUT_DIR:-$ROOT/evaluation/rex_id_val2017}"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
test -f "$ADAPTER/adapter_config.json"
python "$ROOT/rex_token_sft/evaluate_rex_id_coco.py" \
  --model "$MODEL_DIR" --adapter "$ADAPTER" \
  --annotations "$DATA/annotations/instances_val2017.json" --image-dir "$DATA/val2017" \
  --output-dir "$OUTPUT" --batch-size "${EVAL_BATCH_SIZE:-8}"
