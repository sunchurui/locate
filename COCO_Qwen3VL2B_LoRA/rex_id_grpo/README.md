# Rex original-ID LoRA GRPO

Continue `rex_id_sft_outputs_a800_80`, preserving `<0>` through `<999>` at IDs
150643–151642, LoRA rank 16 / alpha 32 and the SFT-trained coordinate embedding
rows. Both the policy and frozen reference load this same SFT adapter. Vision
and projector stay frozen. The unmodified Qwen3-VL-2B weights are referenced by
symlinks in `rex_id_rl_base`; no model shards are duplicated.

## Backend and provenance

Training uses ms-swift 4.6.0 GRPO, not a custom optimization loop. Its `gdpo`
advantage mode normalizes each reward within each eight-response group,
combines with weights 0.7 / 0.3, then normalizes the combined batch advantages.
This matches the dual normalization described by GroundingPI. Rex-Omni's
original EasyR1/veRL branch does not provide this experiment's Qwen3-VL/PEFT
integration; the older patch under `rex_groundingpi_adaptation` is retained as
reference, not the executable training backend here.

The reward plugin imports Rex-Omni's official IoU implementation via
`rex_groundingpi_adaptation/groundingpi_reward.py`, pinned Rex revision
`6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e`. The parser accepts the SFT Rex
object/box markers and dedicated coordinate tokens, preserving them at decode.
Predicted coordinates are rescaled from 0–999 to the annotation's 0–1000 scale.

The set reward follows GroundingPI's best-overlap-then-class-check soft F1.
The strict reward weights format, count, mean F1 at IoU .5/.75/.95, matched IoU,
and within-category x ordering, with duplicate and oversized-box penalties.
Matching is maximum-cardinality and class-aware; duplicate IoU threshold .9,
oversized area threshold .8, count=min/max, and mean IoU over matches at .5
are documented implementation choices where the paper omits exact conventions.
This is an adaptation, not an official GroundingPI release. Empty GT rewards
only empty valid predictions as 1. Invalid output receives 0 for both rewards.

## Reproduce

Prepare the existing fixed manifests and JSONL sources with `prepare_rl_subset.py`
(see the repository's RL data instructions), then:

```bash
bash rex_id_grpo/setup.sh
python rex_id_grpo/prepare_rl.py --root "$PWD"
mkdir -p logs
sbatch rex_id_grpo/train.sbatch
# After a successful complete training run:
sbatch --dependency=afterok:TRAIN_JOB_ID rex_id_grpo/evaluate.sbatch
```

The preparation verifies the existing 10,000 train / 256 validation image IDs
against the saved manifests and changes only the user prompt to the Rex SFT
prompt. Membership uses SHA256 ranking with seed 42; RL training/data order
seed is 20260812. Validation contains no training images.

Default: one GPU with at least 80GB, HF generation, 8 responses per image,
microbatch 1 / accumulation 16 (16 response sequences per optimizer update),
1024 image tokens, 1024 generated tokens, sequence limit 4096; 1 epoch.
Learning rate 5e-7 constant, AdamW .9/.999, weight decay .01, clip 1,
GRPO clip .2, KL beta .02. Validation reward on fixed val256 every 100 updates;
checkpoint every 50 updates. Full COCO AP uses the existing Rex evaluator after
training and is distinct from validation reward.

## Update batch: current recipe versus GroundingPI

GroundingPI section 7.6, Table 11 specifies **32 distinct image–query pairs,
8 responses each, 256 response sequences per optimizer update**. Its 32 is a
prompt count, not 32 time steps or 32 optimizer steps. Its 256 GPUs process
one response per device, with accumulation 1 and one update per rollout.

The current submitted single-GPU recipe instead uses **2 distinct image–query
pairs, 8 responses each, 16 response sequences per optimizer update**:

```
response batch = GPU count * per_device_train_batch_size * gradient_accumulation_steps
               = 1 * 1 * 16 = 16
distinct prompts per update = response batch / num_generations = 16 / 8 = 2
```

With Swift's default generation batching, `steps_per_generation` inherits the
16 accumulation steps, so the rollout contains 16 responses. Those responses
are scored, normalized and split into 16 single-response backward passes;
one optimizer update follows. `num_iterations=1` means no extra optimizer
updates reusing that rollout. Each prompt's GRPO reward-normalization group
still contains eight responses. The final GDPO batch normalization currently
spans 16 responses, rather than the paper's 256.

Thus the current effective response batch is 16 times smaller than the paper;
this is a material adaptation, not an exact training-configuration replication.
The learning rate 5e-7, constant/no-warmup schedule, clip .2, KL .02, response
group size eight and single update per rollout follow the paper. The paper
updates all language parameters, including both vocabulary matrices; this
recipe updates the SFT LoRA and its 1,000 coordinate rows. A learning rate
chosen for full-language training is not established as optimal for LoRA.
No learning-rate or update-batch change was made during the documentation audit.

Matching 256 responses on one GPU requires accumulation across 256 response
microbatches at microbatch 1. Generation can be chunked to save memory, but
the advantage normalization must still span the full 256-response rollout;
merely increasing accumulation while keeping independent 16-response GDPO
normalization batches does not exactly reproduce the paper's update.

Slurm default time limit is 12 hours, not a completion estimate. A timeout can
be resumed by exporting `RESUME_FROM_CHECKPOINT` and the same `OUTPUT_DIR`.
Set `EXPERIMENT_ROOT`, `VENV_PATH`, `SFT_ADAPTER` for another machine. Train only
on >=80GB GPUs. Candidate GPU jobs must use separate `OUTPUT_DIR` values; no
lock or mutex is used. Before keeping one candidate, inspect actual progress
and cancel the other candidate jobs manually. Prior SFT/eval jobs are unrelated.

The environment used for submission has CUDA PyTorch 2.5.1+cu121,
Transformers 4.57.6, PEFT 0.20.0, TRL 0.29.1, and ms-swift 4.6.0.
`setup.sh` assumes the project virtual environment and compatible site CUDA
PyTorch already exist. `run_rlhf.py` supplies the same legacy FSDP import shim
as the SFT launcher. No upstream optimizer/trainer source files are modified.
