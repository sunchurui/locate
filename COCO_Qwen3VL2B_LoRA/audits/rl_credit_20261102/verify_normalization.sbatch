#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=credit_norm_verify
#SBATCH --output=/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/audits/rl_credit_20261102/normalization-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/venv/bin/python -u /home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/rex_id_object_credit/verify_normalization.py --root /home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA
