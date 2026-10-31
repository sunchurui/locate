#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=coco_rex_id_grpo_safe32
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_ROOT/rex_id_grpo_outputs_aligned_safe32}"
export PERFORMANCE_PROFILE=batch32 SAVE_FIRST_STEP=1
unset OBJECT_CREDIT_CONFIG
source "${VENV_PATH:-$EXPERIMENT_ROOT/venv}/bin/activate"
if [[ -d "$OUTPUT_DIR" ]]; then
  checkpoint="$(python "$EXPERIMENT_ROOT/rex_id_grpo/resume_checkpoint.py" "$OUTPUT_DIR")"
  if [[ "$checkpoint" == COMPLETE ]]; then
    printf 'All updates already completed.\n'
    exit 0
  fi
else
  checkpoint="$EXPERIMENT_ROOT/rex_id_grpo_outputs_aligned_fast64/checkpoint-1"
fi
export RESUME_FROM_CHECKPOINT="$checkpoint"
bash "$EXPERIMENT_ROOT/rex_id_grpo/train.sbatch"
