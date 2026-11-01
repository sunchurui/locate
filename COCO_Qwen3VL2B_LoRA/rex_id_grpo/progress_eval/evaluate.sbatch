#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=coco_rex_progress_ap
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$ROOT"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to an SFT adapter or saved RL checkpoint}"
: "${EVAL_OUTPUT_DIR:?Set a separate EVAL_OUTPUT_DIR for this model/step}"
mkdir -p "$EVAL_OUTPUT_DIR"
cp "$ROOT/rl_val_manifest.json" "$EVAL_OUTPUT_DIR/subset_manifest.json"
ADAPTER="$(python rex_id_grpo/progress_eval/prepare_adapter.py \
  --checkpoint "$CHECKPOINT_PATH" --metadata "$ROOT/rex_id_rl_base" \
  --output "$EVAL_OUTPUT_DIR/adapter_view")"
python -u rex_token_sft/evaluate_rex_id_coco.py \
  --model "$ROOT/model" --adapter "$ADAPTER" \
  --annotations "$ROOT/data/coco2017/annotations/instances_val2017.json" \
  --image-dir "$ROOT/data/coco2017/val2017" --image-ids "$ROOT/rl_val_manifest.json" \
  --output-dir "$EVAL_OUTPUT_DIR" --batch-size "${EVAL_BATCH_SIZE:-32}"
