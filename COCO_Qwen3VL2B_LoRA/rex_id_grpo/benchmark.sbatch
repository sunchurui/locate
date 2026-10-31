#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=coco_rex_id_speed
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$EXPERIMENT_ROOT"
export PROFILE_PHASE_TIMES=1 BENCHMARK_UPDATES=2
for profile in batch8 batch16; do
  export PERFORMANCE_PROFILE="$profile"
  export OUTPUT_DIR="$EXPERIMENT_ROOT/speed_benchmarks/${SLURM_JOB_ID}_${profile}"
  printf 'Benchmark %s: same 256 responses per update; two updates\n' "$profile"
  if ! bash "$EXPERIMENT_ROOT/rex_id_grpo/train.sbatch"; then
    printf 'Benchmark profile %s failed; inspect logs. Original training is unaffected.\n' "$profile" >&2
  fi
done
