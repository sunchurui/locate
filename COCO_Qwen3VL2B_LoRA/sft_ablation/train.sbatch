#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=rex_sft_ablation
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
export TMPDIR="$ROOT/runtime_tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
: "${METHOD:?Set METHOD=m34,m37,m38,or m36}"
# Two real gradient updates on the same >=80GB allocation, then fresh formal SFT.
if [[ "${SKIP_SMOKE:-0}" != 1 ]]; then
  python -u sft_ablation/train.py --root "$ROOT" --method "$METHOD" --smoke
fi
args=(--root "$ROOT" --method "$METHOD" --dataloader-workers "${DATALOADER_WORKERS:-4}")
if [[ -n "${OUTPUT_DIR:-}" ]]; then args+=(--output-dir "$OUTPUT_DIR"); fi
python -u sft_ablation/train.py "${args[@]}"
