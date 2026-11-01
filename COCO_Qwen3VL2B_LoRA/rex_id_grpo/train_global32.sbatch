#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=coco_rex_id_global32
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_ROOT/rex_id_grpo_outputs_global32_v2}"
export PERFORMANCE_PROFILE=batch32 SAVE_FIRST_STEP=1
# Start from the same SFT adapter as object-credit training. Do not load the
# old torch2.5 optimizer checkpoint or change the active training environment.
unset OBJECT_CREDIT_CONFIG RESUME_FROM_CHECKPOINT
bash "$EXPERIMENT_ROOT/rex_id_grpo/train.sbatch"
