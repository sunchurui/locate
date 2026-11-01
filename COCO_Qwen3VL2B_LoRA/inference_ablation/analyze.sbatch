#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=rex_inference_compare
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
export OMP_NUM_THREADS=2
exec "$ROOT/venv/bin/python" -u inference_ablation/analyze.py --root "$ROOT"
