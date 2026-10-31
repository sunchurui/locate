#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=coco_rex_id_grpo
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ROOT="$EXPERIMENT_ROOT"
cd "$ROOT"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export IMAGE_MAX_TOKEN_NUM=1024
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset WANDB_DIR
python "$ROOT/rex_id_grpo/check_alignment.py" --require-confirmation
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c 'import torch; props=torch.cuda.get_device_properties(0); print(props); assert props.total_memory >= 79*1024**3, "Training requires an 80GB or larger GPU"'
ADAPTER="${SFT_ADAPTER:-$ROOT/rex_id_sft_outputs_a800_80}"
OUTPUT="${OUTPUT_DIR:-$ROOT/rex_id_grpo_outputs}"
resume=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
python -u "$ROOT/rex_id_grpo/run_rlhf.py" \
  --rlhf_type grpo --model "$ROOT/rex_id_rl_base" --model_type qwen3_vl \
  --adapters "$ADAPTER" --ref_adapters "$ADAPTER" --tuner_type lora \
  --external_plugins "$ROOT/rex_id_grpo/reward_plugin.py" \
  --reward_funcs rex_id_set rex_id_strict --reward_weights 0.7 0.3 \
  --scale_rewards gdpo --advantage_estimator grpo --kl_in_reward false \
  --loss_type grpo --importance_sampling_level token --epsilon 0.2 --epsilon_high 0.2 --beta 0.02 \
  --dataset "$ROOT/data/coco2017/annotations/train_rex_id_grpo_10000.jsonl" \
  --val_dataset "$ROOT/data/coco2017/annotations/val_rex_id_grpo_256.jsonl" \
  --split_dataset_ratio 0 --remove_unused_columns false \
  --num_generations 8 --num_generations_eval 1 --num_iterations 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 256 \
  --generation_batch_size 256 \
  --per_device_eval_batch_size 4 --use_vllm false \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-1024}" --max_length 4096 \
  --temperature 1.0 --top_p 1.0 --top_k 0 \
  --learning_rate 5e-7 --lr_scheduler_type constant --warmup_ratio 0 \
  --num_train_epochs "${NUM_EPOCHS:-1}" --weight_decay 0.01 \
  --optim adamw_torch_fused --adam_beta1 0.9 --adam_beta2 0.999 \
  --adam_epsilon 1e-8 --max_grad_norm 1.0 \
  --freeze_vit true --freeze_aligner true --gradient_checkpointing true \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --seed 20260812 --data_seed 20260812 \
  --eval_strategy steps --eval_steps 100 --save_strategy steps --save_steps 50 \
  --save_total_limit 5 --logging_steps 1 --log_completions true \
  --dataloader_num_workers 4 --dataset_num_proc 4 --report_to none \
  --output_dir "$OUTPUT" --add_version false \
  "${resume[@]}"
