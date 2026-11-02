#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=m30_eval
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
source venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$ROOT/runtime_tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
: "${VARIANT:?Set VARIANT=curriculum or random_control}"
python -u sft_ablation/evaluate.py --root "$ROOT" \
  --adapter "$ROOT/sft_curriculum_outputs/$VARIANT" \
  --output "$ROOT/evaluation/sft_curriculum/$VARIANT" --batch-size 16
python -u rescore_micro.py --root "$ROOT"
python -u inference_ablation/analyze.py --root "$ROOT"
python -u migrate_result_views.py
