#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=coco_qwen2b_lora
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
DATA=$ROOT/data/coco2017
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
cd "$ROOT"
export QWENVL_BBOX_FORMAT=new
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
unset WANDB_DIR

test -f "$DATA/annotations/instances_train2017.json"
test -f "$DATA/annotations/instances_val2017.json"
test -d "$DATA/train2017"
test -d "$DATA/val2017"
test -s "$MODEL_DIR/model.safetensors"

python "$ROOT/prepare_coco_jsonl.py" \
  --instances "$DATA/annotations/instances_train2017.json" \
  --image-dir "$DATA/train2017" \
  --output "$DATA/annotations/train_qwen.jsonl"
python "$ROOT/prepare_coco_jsonl.py" \
  --instances "$DATA/annotations/instances_val2017.json" \
  --image-dir "$DATA/val2017" \
  --output "$DATA/annotations/val_qwen.jsonl"

nvidia-smi
swift sft --model_type qwen3_vl \
  --model "$MODEL_DIR" \
  --dataset "$DATA/annotations/train_qwen.jsonl" \
  --val_dataset "$DATA/annotations/val_qwen.jsonl" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_checkpointing true \
  --max_length 4096 \
  --eval_strategy epoch \
  --save_strategy steps \
  --save_steps 2500 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --output_dir "$OUTPUT_DIR"

ADAPTER=$(find "$OUTPUT_DIR" -type f -name adapter_config.json -printf '%h\n' | sort -V | tail -n1)
test -n "$ADAPTER"
echo "Evaluating adapter: $ADAPTER"
python "$ROOT/evaluate_coco.py" \
  --model "$MODEL_DIR" \
  --adapter "$ADAPTER" \
  --annotations "$DATA/annotations/instances_val2017.json" \
  --image-dir "$DATA/val2017" \
  --output-dir "$ROOT/evaluation/val2017"
echo "Training and COCO val evaluation completed."
