#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=16:00:00
#SBATCH --job-name=coco_ap_checkpoint_ready
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"
: "${CHECKPOINT_STEP:?Set CHECKPOINT_STEP}"
exec "$ROOT/venv/bin/python" -u rex_id_grpo/progress_eval/wait_checkpoint.py \
  --checkpoint "$CHECKPOINT_PATH" --step "$CHECKPOINT_STEP"
