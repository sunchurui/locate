#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --job-name=coco_rex_id_grpo_eval
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/rex_id_grpo_outputs}"
ADAPTER="$(python "$ROOT/rex_id_grpo/resolve_checkpoint.py" "$OUTPUT_DIR" "$ROOT/rex_id_rl_base")"
python "$ROOT/rex_token_sft/evaluate_rex_id_coco.py" \
  --model "$ROOT/model" --adapter "$ADAPTER" \
  --annotations "$ROOT/data/coco2017/annotations/instances_val2017.json" \
  --image-dir "$ROOT/data/coco2017/val2017" \
  --output-dir "${EVAL_OUTPUT_DIR:-$OUTPUT_DIR/coco_val2017}" \
  --batch-size "${EVAL_BATCH_SIZE:-8}"
