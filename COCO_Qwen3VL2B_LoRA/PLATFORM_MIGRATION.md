# Platform migration — 2026-11-03

The user stopped the SCIR queue because this platform has no available quota.
Do not resubmit old SCIR jobs or restore recurring monitoring without explicit
user authorization. Completed models, datasets, predictions and logs remain
on SCIR. Cancellation history is in `migration_20261103/cancelled_queue.json`.

## Code and artifacts

Clone `https://github.com/sunchurui/locate` and use `COCO_Qwen3VL2B_LoRA`.
The code is in GitHub; model weights and COCO images are separate assets.
Before launching the new RL pair, transfer or recreate these paths under the
new experiment root (preserving their relative directory structure):

- `model/`: original Qwen3-VL-2B-Instruct weights and model configuration.
- `rex_id_sft_outputs_5epoch_b64/checkpoint-5547/`: selected epoch3 SFT adapter,
  including adapter weights/config, tokenizer/processor, chat template and
  trainer_state.json. Expected adapter weight SHA256:
  `6432882a8e82f709951f28e52e64678715011deb26015ba5f5e67aecd1414fc2`.
- `data/coco2017/train2017/`, `val2017/` and the original instances annotation
  JSON files under `data/coco2017/annotations/`.
- Keep `rl_train_10000_manifest.json` and `rl_val_manifest.json` from the repo;
  they fix subset membership and source annotation hashes.
- For other SFT experiments, also retain the Rex-ID tokenizer, original
  preparation metadata and exact ordered image-ID manifests.

Do not copy a Python venv or old absolute symlinks as a portable environment.
The measured package versions are in `migration_20261103/environment_versions.json`;
the tested environment used Python3.12 and CUDA PyTorch2.5.1+cu121. Recreate an
environment appropriate to the new host driver, then install the pinned project
dependencies through `setup_env.sh`, `rex_id_grpo/setup.sh` and
`rex_id_object_credit/setup.sh`. Build the isolated evaluator using
`python official_eval/build_counts_engine.py` (see EVALUATION_PROTOCOL.md).

## Rebuild machine-dependent paths on the NEW platform

Existing JSONL image paths and `rex_id_rl_base` symlinks may reference SCIR.
Do not directly launch with those old absolute paths. From the new root:

```bash
export EXPERIMENT_ROOT="$PWD"
source "$PWD/venv/bin/activate"
python prepare_rl_subset.py \
  --instances data/coco2017/annotations/instances_train2017.json \
  --image-dir data/coco2017/train2017 --split train2017 --count 10000 --seed 42 \
  --manifest rl_train_10000_manifest.json \
  --output data/coco2017/annotations/train_rl_10000.jsonl
python prepare_rl_subset.py \
  --instances data/coco2017/annotations/instances_val2017.json \
  --image-dir data/coco2017/val2017 --split val2017 --count 256 --seed 42 \
  --manifest rl_val_manifest.json \
  --output data/coco2017/annotations/val_rl_256.jsonl
python rex_id_grpo/prepare_rl.py --root "$PWD" \
  --sft-adapter "$PWD/rex_id_sft_outputs_5epoch_b64/checkpoint-5547"
python rex_id_object_credit/prepare_metadata.py --root "$PWD"
```

`prepare_rl.py` creates a fresh tokenizer-only base overlay with symlinks to the
new root's original model, plus the fixed 10000-image RL epoch order. Both
policy and reference adapters still come from checkpoint5547; this does not
merge the adapter into the base or initialize from a previous RL result.
Other SFT JSONLs must similarly be regenerated with existing preparation tools
and checked against their frozen image IDs. Preserve original outputs; use a
fresh output directory for any new training run.

## Execution on the replacement platform

Both new epoch3 RL runs were cancelled BEFORE training; there is no new-RL
optimizer state to resume. Launch fresh from the shared epoch3 SFT adapter.
Use `rex_id_epoch3_rl/train.sbatch` with `RL_VARIANT=global` or `credit_norm`.
The scripts retain one epoch/313 updates, fixed data and sampling seed20260812,
32 prompts x8 answers/update, microbatch32 x accumulation8, LR5e-7. The credit
variant uses response-local object z-score and independent stop penalties.

If the replacement platform has Slurm, set its partition/GRES names and use
explicit `EXPERIMENT_ROOT` in the submitted environment. The SCIR GRES names
are site-specific. If it supplies a plain Linux GPU allocation instead, run
the same epoch3 wrapper with Bash; `#SBATCH` lines are comments in this mode:

```bash
# Run only after the new platform/GPU allocation has been selected.
EXPERIMENT_ROOT="$PWD" RL_VARIANT=global \
  CUDA_VISIBLE_DEVICES=0 bash rex_id_epoch3_rl/train.sbatch
# In a separate allocation/session:
EXPERIMENT_ROOT="$PWD" RL_VARIANT=credit_norm \
  CUDA_VISIBLE_DEVICES=0 bash rex_id_epoch3_rl/train.sbatch
```

These foreground commands require a durable session supplied by the platform
or tmux; terminating the shell can terminate training. Each training GPU must
have at least80GB VRAM. Do not start both runs on the same single-GPU allocation.
Submission to the replacement platform has NOT been performed by this migration.

## Evaluation

Use the CURRENT project metric documented in EVALUATION_PROTOCOL.md:
`rex_matching_global_endpoint_micro_v1`. It pools native Rex TP/FP/FN over all
images and categories per IoU; meanIoU F1 is the harmonic mean of the averaged
precision and recall. It is not AP, AP_set, original Rex F1 or GroundingPI GAM.
The epoch3 wrapper evaluates val256 at100/200/300 and saves checkpoints every50.
Use `rex_id_epoch3_rl/evaluate.sbatch` with base/global/credit_norm to evaluate
the shared SFT base and each completed RL model on all5000 val images. On a
non-Slurm host this evaluator can also be run with Bash in an allocated GPU
session. The source/native counts engine must have been built first.

## Verified local checkpoint backup

All 51 files (751481104 bytes total) were downloaded to the user's Windows
workspace under `HPC迁移备份/2026-11-03/checkpoints/` and verified against remote
per-file SHA256 values on 2026-11-03:

| Local directory | Source experiment | Epoch | Purpose |
| --- | --- | --- | --- |
| checkpoint-14786 | rex_id_sft_outputs_a800_80 | 1 | Original one-epoch SFT final |
| checkpoint-9245 | rex_id_sft_outputs_5epoch_b64 | 5 | Five-epoch SFT final |
| checkpoint-5547 | rex_id_sft_outputs_5epoch_b64 | 3 | Selected SFT base for RL |

Each is a complete LoRA training checkpoint, including tokenizer/processor,
optimizer, scheduler and RNG state. They do not contain original 2B base-model
weights. On the new platform place each directory back under its corresponding
source experiment directory above, or explicitly configure the adapter path.
The public GitHub manifest contains hashes/metadata only, never model payloads.
