#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=sft_followup_eval
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$ROOT/runtime_tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
: "${VARIANT:?Set experiment variant or control}"
if [[ "$VARIANT" == control ]]; then
  python -u sft_followups/evaluate.py --root "$ROOT" --adapter "$ROOT/sft_curriculum_outputs/random_control" \
    --output "$ROOT/evaluation/sft_followups/control_highres" --profile highres
else
  python -u sft_followups/evaluate.py --root "$ROOT" --adapter "$ROOT/sft_followup_outputs/$VARIANT" \
    --output "$ROOT/evaluation/sft_followups/${VARIANT}_native" --profile native
  if [[ "$VARIANT" == highres ]]; then
    python -u sft_followups/evaluate.py --root "$ROOT" --adapter "$ROOT/sft_followup_outputs/$VARIANT" \
      --output "$ROOT/evaluation/sft_followups/${VARIANT}_highres" --profile highres
  fi
fi
python -u rescore_micro.py --root "$ROOT"
python -u inference_ablation/analyze.py --root "$ROOT"
python -u migrate_result_views.py
