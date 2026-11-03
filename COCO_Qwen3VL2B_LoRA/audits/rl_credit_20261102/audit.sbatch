#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=rl_credit_audit
#SBATCH --output=/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/audits/rl_credit_20261102/slurm-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2
/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/venv/bin/python -u /home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/audits/rl_credit_20261102/audit.py --root /home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA
