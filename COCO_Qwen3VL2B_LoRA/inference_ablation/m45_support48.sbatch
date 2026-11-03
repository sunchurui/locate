#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=m45_support48
#SBATCH --output=/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/logs/%x_%j.log
set -euo pipefail
ROOT=/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA
cd "$ROOT"
export OMP_NUM_THREADS=1
"$ROOT/venv/bin/python" -u "$ROOT/inference_ablation/m45_support48.py" --root "$ROOT"
