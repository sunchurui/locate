#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=coco_rex_id_grpo_resume
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ROOT="$EXPERIMENT_ROOT"
cd "$ROOT"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/rex_id_grpo_outputs}"
checkpoint="$(python "$ROOT/rex_id_grpo/resume_checkpoint.py" "$OUTPUT_DIR")"
if [[ "$checkpoint" == COMPLETE ]]; then
  printf 'The original job completed all updates; continuation is unnecessary.\n'
  exit 0
fi
export RESUME_FROM_CHECKPOINT="$checkpoint"
printf 'Continuing from %s\n' "$checkpoint"
bash "$ROOT/rex_id_grpo/train.sbatch"
