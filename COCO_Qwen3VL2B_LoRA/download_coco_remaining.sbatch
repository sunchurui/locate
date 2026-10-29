#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=coco_val_ann
#SBATCH --output=logs/coco_val_ann_%j.log
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
source "$ROOT/venv/bin/activate"
python "$ROOT/download_coco_remaining_ranges.py"
