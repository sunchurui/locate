#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=rex_sft_ablation_eval
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${METHOD:?Set METHOD=m34,m37,m38,or m36}"
python -u sft_ablation/evaluate.py --root "$ROOT" \
  --adapter "$ROOT/sft_ablation_outputs/$METHOD" --output "$ROOT/evaluation/sft_ablation/$METHOD" --batch-size 16
