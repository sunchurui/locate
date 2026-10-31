# Rex original-ID LoRA GRPO

**Launch disabled pending user confirmation of unpublished details.** The old
16-response jobs 203833–203836 and their evaluations 203852–203855 were cancelled
before training started. No replacement training jobs have been submitted.
`check_alignment.py --require-confirmation` and the training launcher both
refuse to start while `alignment.json` has `unpublished_details_confirmed=false`.
Do not change that flag without the user's explicit confirmation of the
listed outstanding choices.

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

## Prepare; launch only after outstanding details are confirmed

Prepare the existing fixed manifests and JSONL sources with `prepare_rl_subset.py`
(see the repository's RL data instructions), then:

```bash
bash rex_id_grpo/setup.sh
python rex_id_grpo/prepare_rl.py --root "$PWD"
mkdir -p logs
# These commands remain blocked until the documented choices are confirmed:
sbatch rex_id_grpo/train.sbatch
# After a successful complete training run:
sbatch --dependency=afterok:TRAIN_JOB_ID rex_id_grpo/evaluate.sbatch
```

The preparation verifies the existing 10,000 train / 256 validation image IDs
against the saved manifests and changes only the user prompt to the Rex SFT
prompt. Membership uses SHA256 ranking with seed 42; RL training/data order
seed is 20260812. Validation contains no training images.

Prepared recipe: one GPU with at least 80GB, HF generation, 8 responses per image,
microbatch 1 / accumulation 256 (256 response sequences per optimizer update),
1024 image tokens, 1024 generated tokens, sequence limit 4096; 1 epoch.
Learning rate 5e-7 constant, fused AdamW .9/.999 / epsilon 1e-8, weight decay .01, clip 1,
GRPO clip .2, KL beta .02. Validation reward on fixed val256 every 100 updates;
checkpoint every 50 updates. Full COCO AP uses the existing Rex evaluator after
training and is distinct from validation reward.

## Published optimization alignment

[GroundingPI v2 section 7.6, Table 11](https://arxiv.org/html/2609.39601v2#S7.SS6) specifies **32 distinct image–query pairs,
8 responses each, 256 response sequences per optimizer update**. Its 32 is a
prompt count, not 32 time steps or 32 optimizer steps. Its 256 GPUs process
one response per device, with accumulation 1 and one update per rollout.

The corrected single-GPU recipe uses **32 distinct image–query pairs,
8 responses each, 256 response sequences per optimizer update**:

```
response batch = GPU count * per_device_train_batch_size * gradient_accumulation_steps
               = 1 * 1 * 256 = 256
distinct prompts per update = response batch / num_generations = 256 / 8 = 32
```

`generation_batch_size=256` is explicit. The HF engine generates in memory
chunks of 16, collects all 256 answers, then computes both reward components.
Within each eight-response group, each component is normalized with sample
standard deviation and epsilon 1e-8. The weighted .7/.3 sum is then normalized
over all 256 responses. Swift splits those results into 256 one-response
backward passes, followed by one optimizer update. `num_iterations=1` prevents
further parameter updates reusing the rollout. Chunking does not update the
policy or normalize advantages between chunks.

The launcher validates the full parsed configuration before model loading.
It also checks that the actual reward tensor has shape [256,2], covers 32
different image IDs, and contains eight contiguous answers for each image.
It refuses to silently use another microbatch, update batch or objective.

| Published setting | Prepared implementation |
| --- | --- |
| 32 prompts, 8 responses each, 1 update per rollout | Explicit 256-response rollout and accumulation 256 on one GPU |
| Learning rate / schedule | 5e-7 / constant / no warmup |
| Optimizer | Fused AdamW, betas .9/.999, epsilon 1e-8 |
| Weight decay / gradient clipping | .01 / 1 |
| GRPO clipping / reference KL weight | .2 / .02 |
| Loss reduction | Per-response token mean, then response mean |
| Reference KL | Paper equation 1: exp(log-ref minus log-policy) minus that log ratio minus 1 |
| Rewards / normalization | .7/.3, per-component within-prompt then full-response-batch standardization |
| Reference policy | Frozen original-ID SFT adapter, including coordinate rows |
| Frozen modules / recomputation | Vision and projector frozen / language activation checkpointing |
| Precision / seed / data seed | BF16 / 20260812 / 20260812 |

Swift 4.6.0 internally caps sampled KL at 10. `run_rlhf.py` makes a small,
source-checked runtime adaptation replacing only those two KL expressions
with the paper's uncapped formula. It also sets the HF engine's existing
generation batch cap to 16. The upstream rollout, optimizer and training
loops are used. Both adaptations fail for review if the expected source/API
changes; site-package files are not overwritten.

The paper uses 256 GPUs with accumulation 1 and ZeRO-2; one GPU uses equivalent
response-count accumulation instead. The model, LoRA rather than full language
updates, Rex original-ID tokenizer, COCO subset and >=80GB requirement are
explicit user constraints. They are not claimed to reproduce GroundingPI's
model architecture, data or trainable parameter set.

## Unpublished details: launch remains disabled

`alignment.json` separates published optimization, user constraints, physical
execution settings and unresolved choices. The pending choices include strict
count agreement and matching definitions, mean matched IoU, duplicate threshold
and normalization, oversized-box definition, parser/error gating, empty GT,
the ordering score's partial-credit rules, generation sampling/length limits and training
duration. The 10,000-image subset is not divisible by the 32-prompt batch;
treatment of the last 16 prompts also awaits confirmation. Existing adapter
reward code is retained for review; its approximations are not approved as
GroundingPI-equivalent. No unverified reward component is being trained.
The within-label x1 sorting convention itself is explicitly published in
section 7.1 and matches the Rex SFT; it is not an unresolved ordering direction.

Validation cadence, checkpoint cadence and the Slurm time limit are operational
experiment settings; no GroundingPI source is claimed for those numbers.

Slurm default time limit is 12 hours, not a completion estimate. A timeout can
be resumed by exporting `RESUME_FROM_CHECKPOINT` and the same `OUTPUT_DIR`.
Set `EXPERIMENT_ROOT`, `VENV_PATH`, `SFT_ADAPTER` for another machine. Train only
on >=80GB GPUs. Candidate GPU jobs must use separate `OUTPUT_DIR` values; no
lock or mutex is used. Before keeping one candidate, inspect actual progress
and cancel the other candidate jobs manually. Prior SFT/eval jobs are unrelated.

The environment used for submission has CUDA PyTorch 2.5.1+cu121,
Transformers 4.57.6, PEFT 0.20.0, TRL 0.29.1, ms-swift 4.6.0 and msgspec 0.22.0.
`setup.sh` assumes the project virtual environment and compatible site CUDA
PyTorch already exist. `run_rlhf.py` supplies the same legacy FSDP import shim
as the SFT launcher. Upstream files remain unchanged on disk; the small runtime
adaptations and their reasons are described above.
