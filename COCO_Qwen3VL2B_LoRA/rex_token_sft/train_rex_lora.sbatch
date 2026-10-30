#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=coco_rex_token_lora
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA="$ROOT/data/coco2017"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
REX_SOURCE="${REX_SOURCE:-$ROOT/references/Rex-Omni}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/rex_token_sft_outputs}"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
unset WANDB_DIR
test -f "$DATA/annotations/train_rex_token.jsonl"
test -f "$DATA/annotations/val_rex_token.jsonl"
test -f "$REX_SOURCE/finetuning/dataset/task_fns/grounding_task.py"
test -f "$MODEL_DIR/config.json"

resume=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi
python "$ROOT/rex_token_sft/train_rex_qwen3_lora.py" \
  --model "$MODEL_DIR" --train-data "$DATA/annotations/train_rex_token.jsonl" \
  --val-data "$DATA/annotations/val_rex_token.jsonl" --output-dir "$OUTPUT_DIR" \
  --train-batch-size "${TRAIN_BATCH_SIZE:-4}" --eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --grad-accum "${GRAD_ACCUM:-2}" --epochs "${NUM_EPOCHS:-1}" \
  --learning-rate "${LEARNING_RATE:-0.0001}" --eval-steps 1000 --save-steps 1000 \
  "${resume[@]}"
