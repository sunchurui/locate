# M30-inspired paired curriculum order experiment

User authorized2026-11-02. Same COCO train118287, Qwen3-VL-2B, rank16/alpha32
language LoRA and trainable original coordinate IDs150643..151642. Serializer,
target strings, image-first prompt and all GT remain unchanged. Fresh model/
LoRA initialization, seed42 before PEFT; no old SFT/RL adapter loaded.

## Scope and source

[WeVisOne-VL section4.1](https://arxiv.org/html/2610.31329v1#S4.SS1) first mainly
uses <=50 boxes per class, then allows <=500 per answer while changing visual
budget/context/unfreezing. Our COCO annotations, excluding crowd/invalid boxes,
have at most27/class and90/image. Exact50/500 filtering would change nothing.

This is an explicit **density-order-only adaptation**, not a full reproduction.
The fallback was fixed before validation results: if <10% images exceed50/class,
use training-only75th percentile of max per-class object count. It is5 here:
89385 sparse images (<=5) and28902 dense images (>5).

- Phase1:59136 sparse images, shuffled with seed42.
- Phase2:remaining30249 sparse plus28902 dense, mixed with seed43.
- Every image appears exactly once, all annotations/answers retained.
- Random control:all118287 images shuffled with seed42, same budget and modules.
- SequentialSampler enforces both frozen orders; Trainer must not reshuffle them.
- Phase boundary step7392; total14786 updates, effective batch8 (4x2), one epoch.
- Existing LR1e-4/cosine/3%warmup/weight-decay.01/BF16/clip1 unchanged.
- Visual budget16..1024, context4096, same as existing Rex baseline.
- Validation loss every1000, actual fixed256 generation at5000/10000, full5000
  after successful training. Current global endpoint micro P/R/F1 only.
- Save all checkpoints; each GPU allocation performs2 real smoke updates first.
  Formal starts again from fresh seeded weights. Both require>=80GB GPU.

## Reproduce

```bash
venv/bin/python sft_curriculum/prepare.py --root "$PWD"
sbatch --export=ALL,EXPERIMENT_ROOT="$PWD",VARIANT=curriculum sft_curriculum/train.sbatch
sbatch --export=ALL,EXPERIMENT_ROOT="$PWD",VARIANT=random_control sft_curriculum/train.sbatch
sbatch --dependency=afterok:TRAIN_JOB --export=ALL,EXPERIMENT_ROOT="$PWD",VARIANT=curriculum sft_curriculum/evaluate.sbatch
```

Input/source hashes and exact ordered image IDs are in prepared/report.json and
prepared/*_ids.json. Prepared image paths are regenerated per machine. Comparison
must use the new random control, not only the historical seed-unrecorded model.
Check initial trainable hashes match between the two formal runs when they start.

GPU jobs are in jobs.json. No locks, no duplicate training candidates, no recurring
monitor. Never delete models/data/checkpoints. Generated training rows excluded
from Git; frozen image-ID manifests and preparation metadata are included.


## Startup path recovery (2026-11-03)

Original random-control207966 failed before model initialization: --chdir did
not change SLURM_SUBMIT_DIR, which still pointed to the submission checkout.
Launchers now honor PWD or explicit EXPERIMENT_ROOT and activate the absolute
venv path. Replacements208920/208922 and full evaluators208921/208923 preserve
all data, learning parameters and old logs. Runtime history is in jobs.json.
