#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --job-name=credit_weight_sweep
#SBATCH --output=/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/audits/rl_credit_20261102/weight-sweep-%j.log
set -euo pipefail
/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/venv/bin/python -u /home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/audits/rl_credit_20261102/weight_sweep.py --root /home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA
