#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=coco_qwen2b_grpo
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
DATA="$ROOT/data/coco2017"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
SFT_ADAPTER="${SFT_ADAPTER:-$ROOT/checkpoints/v2-20261029-191142/checkpoint-14786}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/grpo_checkpoints}"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
cd "$ROOT"
export QWENVL_BBOX_FORMAT=new IMAGE_MAX_TOKEN_NUM=1024 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled
unset WANDB_DIR

test -f "$SFT_ADAPTER/adapter_config.json"
python "$ROOT/prepare_rl_subset.py" --instances "$DATA/annotations/instances_train2017.json" \
  --image-dir "$DATA/train2017" --split train2017 --count 5000 --seed 42 \
  --manifest "$ROOT/rl_train_manifest.json" --output "$DATA/annotations/train_rl_5000.jsonl"
python "$ROOT/prepare_rl_subset.py" --instances "$DATA/annotations/instances_val2017.json" \
  --image-dir "$DATA/val2017" --split val2017 --count 256 --seed 42 \
  --manifest "$ROOT/rl_val_manifest.json" --output "$DATA/annotations/val_rl_256.jsonl"

resume=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
python "$ROOT/compat/run_swift_rlhf.py" --rlhf_type grpo --model_type qwen3_vl \
  --model "$MODEL_DIR" --tuner_type lora --adapters "$SFT_ADAPTER" --ref_adapters "$SFT_ADAPTER" \
  --dataset "$DATA/annotations/train_rl_5000.jsonl" --val_dataset "$DATA/annotations/val_rl_256.jsonl" \
  --external_plugins "$ROOT/coco_grpo_reward.py" --reward_funcs coco_detection \
  --torch_dtype bfloat16 --use_vllm false --num_generations 4 --num_generations_eval 4 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 \
  --num_train_epochs 1 --learning_rate 1e-5 --beta 0.01 --temperature 0.9 \
  --max_completion_length 768 --max_length 4096 --truncation_strategy delete \
  --gradient_checkpointing true --seed 42 --data_seed 42 --dataset_num_proc 1 --dataloader_num_workers 0 \
  --eval_strategy steps --eval_steps 100 --save_strategy steps --save_steps 100 --save_total_limit 3 \
  --logging_steps 5 --log_completions true --report_to none --output_dir "$OUTPUT_DIR" "${resume[@]}"
