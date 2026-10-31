#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=coco_rex_id_object_credit
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_ROOT/rex_id_grpo_outputs_object_credit_v1}"
export OBJECT_CREDIT_CONFIG="$EXPERIMENT_ROOT/rex_id_object_credit/config.json"
export PERFORMANCE_PROFILE=batch32
export SAVE_FIRST_STEP=1
source "${VENV_PATH:-$EXPERIMENT_ROOT/venv}/bin/activate"
python "$EXPERIMENT_ROOT/rex_id_object_credit/prepare_metadata.py" --root "$EXPERIMENT_ROOT"
bash "$EXPERIMENT_ROOT/rex_id_grpo/train.sbatch"
