#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=16:00:00
#SBATCH --job-name=coco_progress_ap_submit
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
exec "$ROOT/venv/bin/python" -u rex_id_grpo/progress_eval/watch.py \
  --root "$ROOT" --config "${WATCH_CONFIG_PATH:-$ROOT/progress_eval_jobs.json}"
