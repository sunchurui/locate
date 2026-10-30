#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=coco_qwen2b_eval
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
DATA="$ROOT/data/coco2017"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
ADAPTER="${ADAPTER:-$ROOT/checkpoints/v2-20261029-191142/checkpoint-14786}"
OUT="$ROOT/evaluation/val2017"
BASE_OUT="$ROOT/evaluation/base_val2017_first1600"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
NUM_SHARDS="$NUM_GPUS"
BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
cd "$ROOT"
export QWENVL_BBOX_FORMAT=new
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python "$ROOT/evaluate_coco.py" \
  --model "$MODEL_DIR" \
  --annotations "$DATA/annotations/instances_val2017.json" \
  --image-dir "$DATA/val2017" \
  --output-dir "$BASE_OUT" \
  --limit 1600 --batch-size "$BATCH_SIZE"

python "$ROOT/split_coco_predictions.py" \
  --annotations "$DATA/annotations/instances_val2017.json" \
  --predictions "$OUT/predictions.jsonl" \
  --output-dir "$OUT" \
  --num-shards "$NUM_SHARDS"

PIDS=()
for ((rank=0; rank<NUM_GPUS; rank++)); do
  CUDA_VISIBLE_DEVICES="$rank" python "$ROOT/evaluate_coco.py" \
    --model "$MODEL_DIR" --adapter "$ADAPTER" \
    --annotations "$DATA/annotations/instances_val2017.json" \
    --image-dir "$DATA/val2017" --output-dir "$OUT" \
    --prediction-file "predictions.rank${rank}.jsonl" \
    --shard-index "$rank" --num-shards "$NUM_SHARDS" --batch-size "$BATCH_SIZE" \
    --skip-coco-eval > "$OUT/eval_rank${rank}.log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

python "$ROOT/merge_coco_predictions.py" \
  --annotations "$DATA/annotations/instances_val2017.json" \
  --input-dir "$OUT" --output "$OUT/predictions.jsonl" \
  --num-shards "$NUM_SHARDS"
python "$ROOT/evaluate_coco.py" \
  --model "$MODEL_DIR" --adapter "$ADAPTER" \
  --annotations "$DATA/annotations/instances_val2017.json" \
  --image-dir "$DATA/val2017" --output-dir "$OUT" \
  --skip-inference
echo "Resumed COCO validation and evaluation completed."
