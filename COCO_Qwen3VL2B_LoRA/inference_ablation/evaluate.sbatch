#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=rex_inference_ablation
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${METHOD:?Set METHOD=m06,m44,m45,or m46}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
python -u inference_ablation/run.py --root "$ROOT" --method "$METHOD" \
  --shard-index "$SHARD" --num-shards "${NUM_SHARDS:-1}" \
  --output "$ROOT/inference_ablation/results/$METHOD/shard$SHARD"
