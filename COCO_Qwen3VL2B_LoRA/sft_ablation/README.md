# Independent M34 / M37 / M38 / M36 Rex-ID LoRA SFT experiments

These are four separate one-factor experiments, not a combined recipe. M17 is
owned by another agent and is untouched. All runs start from the same original
Qwen3-VL-2B-Instruct weights in `model/`, with fresh rank16/alpha32/dropout0.05
language-projection LoRA and trainable original Rex coordinate IDs150643–151642.
No existing SFT or RL adapter is loaded into these new training runs.

`baseline_train.py` is a byte-for-byte copy of the established SFT entrypoint,
verified by SHA256 in config. `train.py` reuses its `main`, PEFT construction,
dataset loader, optimizer, loss, masking, image budget and Trainer loop. Only
the data collator and callbacks are supplied by this package. Original code,
models, datasets and other agents' files are not modified.

## Shared protocol

- The complete118287 COCO2017 train image pool, same crowd/invalid-box exclusions.
- The original5000 val images; no train/val mixing or generated extra images.
- One epoch,14786 optimizer updates, microbatch4, accumulation2, effective batch8.
- LR1e-4, cosine,3%warmup, AdamW, weight decay0.01, BF16, gradient checkpointing,
  seed/data-seed42,4096 context and original1024 visual-token pixel budget.
- Set seed42 before creating PEFT so all four new initial adapters are equal.
  Save a SHA256 of initial trainable tensors to verify this. The historical
  entrypoint only sets seeds inside Trainer, so its pre-Trainer LoRA RNG state
  was not explicitly recorded; do not claim bitwise identical initialization
  to the historical completed run. The pretrained base weights are the same.
- Validation loss every1000 updates; save every1000 with **no checkpoint deletion**.
- At5000/10000 updates, run greedy COCO AP on the existing fixed256-image
  manifest within the allocated training GPU, using the in-memory policy.
- Full5000-image COCO AP after successful final training through `afterok`.
  All evaluations use the original q/999 decoder and generated-order scores.
  M37 uses text-first both for validation and final inference. No image
  augmentation or position reweighting is applied at validation/test time.
- Formal training and two-step smoke training both require >=80GB GPU memory.
  CPU/RAM requests are execution settings, not changes to the learning recipe.

## Frozen data variants

`prepare.py` first regenerates every baseline target through the actual pinned
Rex formatter git blob and checks exact agreement with the established JSONL.
It saves source/data hashes, explicit randomness, sampling indices and prepared
JSONLs. Preparation is idempotent only for identical configuration and hashes.

### M34: multi-phrase query sampling

Use [Rex-Omni §4.1](https://arxiv.org/html/2510.12798v1#S4.SS1): randomly choose
1..N existing category phrases, and retain **all** instances of the chosen
classes. Keep image-first order and the official box serializer. In this COCO
adaptation,50% of images retain their original all80-category query,50% receive
a category-subset query (empty images retain their original query). The mixing
ratio is our fixed convention to retain the benchmark task, not a paper default.
This tests textual multi-phrase sampling only; it does not add the separate
visual-example prompting task. Each image's choices are SHA256-seeded from
seed42/method/imageID, fixed for the one-epoch comparison.

### M37: input order

Use [Phi-Ground §5.1](https://arxiv.org/html/2507.23779#S5.SS1). Change only the
user content to `[text, image]` in SFT, validation loss and inference. Targets,
image pixels and categories are unchanged. The established Rex SFT and public
Rex detection entrance use `[image, text]`; this is an explicit ablation.

### M38: random resize and padding

Port [Phi-Ground Algorithm2](https://arxiv.org/html/2507.23779#S5.SS2). Resize
probability1, max_screen_size4096, white canvas, random placement. COCO canvas
dimensions equal original image W,H, preserving processor resolution budgets;
Smax=1 and Smin=min(1,W/4096). Integer resized dimensions and offsets are frozen
per image. Use actual integer x/y scale factors to transform **every GT corner**,
then invoke the pinned Rex quantizer/serializer. No cropped or removed objects,
no synthetic image files, no extra random cropping, no changes to val images.
PIL BICUBIC is an explicit interpolation convention. M38 is a natural-image
adaptation of the screenshot method; very small boxes may collapse to equal
coordinate bins under existing quantization, and their targets are retained.

### M36: position-balanced whole-image resampling

[Phi-Ground Algorithm3](https://arxiv.org/html/2507.23779#A4.SS2) groups target
centers in50x50 cells and caps cell occupancy at a selected quantile. Its
original single-target sampling cannot be used to drop only some annotations
from an all-object COCO answer. Our multi-object adaptation computes
`K=sorted(all2500 cell counts)[floor(2500*0.5)]` and per-center retention
`min(1,K/n_cell)`, then averages over centers to weight each **whole image**.
Empty images receive the mean nonempty-image weight, preserving their expected
fraction. Seeded weighted replacement draws exactly118287 complete examples.
Every drawn image retains **all original GT**; duplicates/undrawn images are
the intended sampling change. Factor0.5, image averaging and equal-budget
replacement are explicit adaptation choices, not reported author defaults.
The saved report includes original/expected/realized center histograms and
entropy/KL, weights, draw indices and unique-image counts.

## Run

The original SFT environment is reused without package changes.

```bash
cd /path/to/COCO_Qwen3VL2B_LoRA
venv/bin/python sft_ablation/prepare.py --root "$PWD"
sbatch --export=ALL,METHOD=m34 sft_ablation/train.sbatch
sbatch --dependency=afterok:TRAIN_JOB --export=ALL,METHOD=m34 sft_ablation/evaluate.sbatch
# Repeat for m37,m38,m36. Override typed GPU GRES only with >=80GB for training.
```

Each training allocation first completes two real smoke updates on a distinct
8-example prepared dataset, then starts formal SFT freshly from the original
base weights. Smoke weights never initialize formal training. The smoke and
formal initial hashes must agree for a method. Source snapshots, saved protocol,
full prepared-file hashes and logs make each change reviewable. A completed
training directory cannot be overwritten by this launcher.

`prepared/`, model outputs and `jobs.json` are runtime artifacts, not committed
to GitHub. Regenerate prepared data from the same inputs/config on another
machine. No locking or mutual exclusion is added to training jobs.
