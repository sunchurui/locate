#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=m30_density
#SBATCH --output=/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/m30_curriculum/density-%j.log
set -euo pipefail
/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/venv/bin/python -u /home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/m30_curriculum/inspect_density.py --root /home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA
