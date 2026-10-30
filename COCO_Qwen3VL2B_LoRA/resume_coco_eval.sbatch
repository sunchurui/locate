#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100-sxm4-80gb:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=coco_qwen2b_eval
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
DATA="$ROOT/data/coco2017"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
ADAPTER="${ADAPTER:-$ROOT/checkpoints/v2-20261029-191142/checkpoint-14786}"
OUT="$ROOT/evaluation/val2017"
NUM_SHARDS=2
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
cd "$ROOT"
export QWENVL_BBOX_FORMAT=new
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python "$ROOT/split_coco_predictions.py" \
  --annotations "$DATA/annotations/instances_val2017.json" \
  --predictions "$OUT/predictions.jsonl" \
  --output-dir "$OUT" \
  --num-shards "$NUM_SHARDS"

CUDA_VISIBLE_DEVICES=0 python "$ROOT/evaluate_coco.py" \
  --model "$MODEL_DIR" --adapter "$ADAPTER" \
  --annotations "$DATA/annotations/instances_val2017.json" \
  --image-dir "$DATA/val2017" --output-dir "$OUT" \
  --prediction-file predictions.rank0.jsonl --shard-index 0 --num-shards "$NUM_SHARDS" --batch-size 4 \
  --skip-coco-eval > "$OUT/eval_rank0.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python "$ROOT/evaluate_coco.py" \
  --model "$MODEL_DIR" --adapter "$ADAPTER" \
  --annotations "$DATA/annotations/instances_val2017.json" \
  --image-dir "$DATA/val2017" --output-dir "$OUT" \
  --prediction-file predictions.rank1.jsonl --shard-index 1 --num-shards "$NUM_SHARDS" --batch-size 4 \
  --skip-coco-eval > "$OUT/eval_rank1.log" 2>&1 &
PID1=$!
wait "$PID0"
wait "$PID1"

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
