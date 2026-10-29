#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=coco_download
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
source "${VENV_PATH:-$ROOT/venv}/bin/activate"
bash "$ROOT/finish_coco_modelscope.sh"
