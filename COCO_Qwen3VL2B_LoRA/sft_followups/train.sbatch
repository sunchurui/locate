#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=sft_followup
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$ROOT/runtime_tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
: "${VARIANT:?Set image_mean, projector_lora, highres, or class_rfs}"
python -u sft_followups/train.py --root "$ROOT" --variant "$VARIANT" --smoke \
  --output "$ROOT/sft_followup_outputs/smoke_${VARIANT}_${SLURM_JOB_ID}"
python -u sft_followups/train.py --root "$ROOT" --variant "$VARIANT" \
  --output "$ROOT/sft_followup_outputs/$VARIANT"
