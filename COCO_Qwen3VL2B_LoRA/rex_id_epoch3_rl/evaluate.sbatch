#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --job-name=rex_epoch3_rl_eval
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "${RL_VARIANT:?Set base/global/credit_norm}" in
 base) ADAPTER="$ROOT/rex_id_sft_outputs_5epoch_b64/checkpoint-5547" ;;
 global) ADAPTER="$(python rex_id_grpo/resolve_checkpoint.py "$ROOT/rex_id_grpo_outputs_epoch3_global_v1" "$ROOT/rex_id_rl_base")" ;;
 credit_norm) ADAPTER="$(python rex_id_grpo/resolve_checkpoint.py "$ROOT/rex_id_grpo_outputs_epoch3_credit_norm_v2" "$ROOT/rex_id_rl_base")" ;;
 *) exit 2 ;;
esac
python -u rex_token_sft/evaluate_rex_id_coco.py \
 --model "$ROOT/model" --adapter "$ADAPTER" \
 --annotations "$ROOT/data/coco2017/annotations/instances_val2017.json" \
 --image-dir "$ROOT/data/coco2017/val2017" \
 --output-dir "$ROOT/evaluation/epoch3_rl/${RL_VARIANT}_full" --batch-size 16
