#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=m30_sft
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
# --chdir sets PWD; SLURM_SUBMIT_DIR can still name a staging checkout.
ROOT="${EXPERIMENT_ROOT:-$PWD}"
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$ROOT/runtime_tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
: "${VARIANT:?Set VARIANT=curriculum or random_control}"
python -u sft_curriculum/train.py --root "$ROOT" --variant "$VARIANT" --smoke \
  --output "$ROOT/sft_curriculum_outputs/smoke_${VARIANT}_${SLURM_JOB_ID}"
python -u sft_curriculum/train.py --root "$ROOT" --variant "$VARIANT" \
  --output "$ROOT/sft_curriculum_outputs/$VARIANT"
