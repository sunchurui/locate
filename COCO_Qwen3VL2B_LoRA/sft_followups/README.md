# Four source-grounded Rex-ID SFT ablations

Authorized2026-11-02: image-balanced loss, visual-projector LoRA, actual high
resolution and class-balanced whole-image sampling. All are INDEPENDENT runs.
Qwen3-VL-2B, original coordinate IDs150643..151642, rank16/alpha32/dropout.05,
COCO train pool118287/val5000, seed42, image-first all80-class prompt, exact Rex
answers. No external data. No existing SFT/RL adapter initialization.

## Existing work and adaptation boundary

| Run | Primary evidence | What this project changes |
| --- | --- | --- |
| image_mean | [Nemotron3 Super report section3.1 pp18-19](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf), [official SFT equations](https://docs.nvidia.com/nemotron/latest/nemotron/super3/sft.html) | Uses their sample-average formula for the entire single-factor run, one image=one sample. Not their two-stage reasoning/long-context recipe. No claimed preexisting COCO improvement |
| projector_lora | [LLaVA LoRA recipe](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh), [LLaVA training](https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py), [Swift freeze_aligner documentation](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md) | Adds LoRA to Qwen's main merger and3 DeepStack mergers (8 linears), freezes vision blocks. LLaVA trains projector weights; low-rank projector instead is our specific adaptation, also supported by Swift's aligner-LoRA route |
| highres | [WeVisOne-VL section4.1](https://arxiv.org/html/2610.31329v1#S4.SS1) | Isolates actual resolution; doesn't unfreeze vision or change context/data curriculum. Forces roughly1024..1280 merged visual tokens using actual patch16*merge2=32, not just a larger inactive pixel cap |
| class_rfs | [LVIS](https://arxiv.org/abs/1908.03195), [Detectron2 original sampler](https://github.com/facebookresearch/detectron2/blob/247691a1c292654867deb35cabb27f18d75a1363/detectron2/data/samplers/distributed_sampler.py) | Original factor computation AND stochastic-repeat/shuffle stream directly extracted from pinned AST. Fix first118287 draws for equal updates; adapt threshold from training statistics if LVIS threshold gives negligible change on COCO |

Only the method component is borrowed. We do NOT claim to reproduce these full
papers or their reported gains. Source hashes/license for the directly reused
Detectron2 class are in sources.lock.json/upstream/ (Apache2).

## Shared control and budget

Control is the already submitted M30 random-order run207966, evaluated natively
by207967. Non-resampling variants use its EXACT frozen image order. Preserve
its initialization and optimizer schedule: micro4 x accumulation2=effective8,
14786updates, one epoch's draws, LR1e-4 cosine/3%warmup, weight-decay.01, BF16,
clip1, context4096, language projections and tied original coordinate-token rows.

Projector extension normally changes language LoRA RNG consumption. core.py
first constructs the exact standard seeded adapter, saves all fresh common
trainables, unloads WITHOUT merging, builds the extended adapter, and restores
the common tensors verbatim. It verifies the hash and only permits16 extra
A/B tensors in8 merger linears. Vision backbone remains frozen.

## Image-average loss and accumulation

L=(1/N_images) sum_i [(1/n_output_tokens_i) sum_t CE_it]. Only assistant tokens
participate, after causal shifting. The divisor is the number of IMAGES across
the entire accumulation window, including a final partial window. Trainer's
custom loss hook skips extra accumulation division. This is checked against
real Trainer updates with micro3/acc1 versus micro1/acc3 and unequal lengths,
including the partial tail. Don't substitute a microbatch mean without checking.

## RFS details

f_c=fraction of original training images containing category c. r_c=max(1,sqrt(t/f_c));
r_i=max_{c in image_i} r_c (empty images r_i=1). Include every eligible GT on each
draw. Fixed seed42, original Detectron2 stochastic rounding and shuffled stream.
Use t=.001 if it affects>=5% images, otherwise lower median of the80 training
category frequencies. This fallback and the fixed draw budget are COCO-specific
adaptations, chosen before any validation scores. Unique images seen may fall
below118287 because rare-category images repeat; original pool is unchanged.

## Validation and crossed resolution

Loss validation every1000 steps; generation on fixed val256 at5000/10000.
All primary progress/final comparisons use NATIVE inference resolution. For
highres SFT additionally generate highres predictions. A separate afterok job
evaluates the random control at highres too, completing a2x2 comparison so
training and inference resolution gains are not conflated. No adapter files
are modified when overriding an evaluation profile.

Scoring is rex_matching_global_endpoint_micro_v1 everywhere; no confidence AP.
Full5000 results run after successful training and refresh current result views.
All checkpoints retained. Every allocation first performs2 real gradient smoke
updates (including nonzero projector gradients), then starts fresh formal SFT.
Training/smoke require>=80GB; inference uses BF16 Ampere/newer.

```bash
venv/bin/python sft_followups/prepare.py --root "$PWD"
venv/bin/python sft_followups/check.py
sbatch --export=ALL,VARIANT=image_mean sft_followups/train.sbatch
sbatch --dependency=afterok:TRAIN_JOB --export=ALL,VARIANT=image_mean sft_followups/evaluate.sbatch
```

Exact manifests/hashes: prepared/report.json and *_ids.json. Local absolute image
paths are regenerated per machine. Jobs are in jobs.json; preparation/preflight
CPU jobs in cpu_jobs.json. No locks or recurring Codex monitors are introduced.
