#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --job-name=rex_epoch3_rl
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$SLURM_SUBMIT_DIR}"
ROOT="$EXPERIMENT_ROOT"
: "${RL_VARIANT:?Set RL_VARIANT=global or credit_norm}"
export SFT_ADAPTER="$ROOT/rex_id_sft_outputs_5epoch_b64/checkpoint-5547"
export PERFORMANCE_PROFILE=batch32 SAVE_FIRST_STEP=1 NUM_EPOCHS=1
export RLHF_RUNNER="$ROOT/rex_id_epoch3_rl/run_pair.py"
unset RESUME_FROM_CHECKPOINT BENCHMARK_UPDATES OBJECT_CREDIT_CONFIG
case "$RL_VARIANT" in
 global) export OUTPUT_DIR="$ROOT/rex_id_grpo_outputs_epoch3_global_v1" ;;
 credit_norm)
   export OUTPUT_DIR="$ROOT/rex_id_grpo_outputs_epoch3_credit_norm_v2"
   export OBJECT_CREDIT_CONFIG="$ROOT/rex_id_object_credit/config_mcr_norm.json" ;;
 *) exit 2 ;;
esac
bash "$ROOT/rex_id_grpo/train.sbatch"
