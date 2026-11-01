# Validation before GPU allocation (2026-10-31)

The new implementation is submitted as training job **204084**, requesting
one A100 SXM 80GB, with full COCO evaluator **204088** depending on success.
The repaired global-reward baseline is job **204091**, requesting one A100
PCIe 80GB, with evaluator **204080**. These were pending at the last check;
submission does not mean GPU training or backward has been verified.

Completed checks:

- Python syntax and Slurm/Bash syntax.
- Import through the real patched ms-swift training CLI (`run_rlhf.py --help`).
- All execution profiles retain 256 responses per optimizer update.
- Original train10000 and val256 annotation hashes and IDs: unchanged.
  Crowd/ignore metadata contains 855 train and 22 validation regions.
- Actual original-ID tokenizer validates all 1000 coordinate IDs and the
  single-token Rex structural markers.
- Audit of all three saved complete global-reward rollouts: 768 real decoded
  responses, 758 strictly valid, 5556 parsed boxes, 3779 one-to-one matched,
  1500 unmatched evaluable and 277 crowd-ignored. Credit is bounded and each
  box points to exactly four coordinate tokens; no GT is matched twice.
  2435 boxes receive positive and 2844 negative credit corrections.

The saved completions do not contain original generated IDs or termination
reasons. The offline audit retokenizes real text and assumes normal stopping
for prospective class-stop annotations (444 sites). It does not synthesize an
EOS or claim this establishes alignment to actual generation/loss masks.

At runtime, `OBJECT_CREDIT_VERIFIED` is printed only after the complete actual
32x8 rollout has passed token-ID/loss-mask checks. The audit file records actual
sampled IDs, object matches and stop sites. First backward and checkpoint
creation must still be confirmed after Slurm allocates a GPU.

The former batch64 baseline job 204013 failed at update 4 with OOM, after three
updates and a complete checkpoint-1. Recovery uses microbatch32/accumulation8,
preserves that checkpoint and all old outputs, and has no checkpoint retention
limit. The object-credit variant starts independently from the same SFT adapter
used to initialize the original baseline, rather than from its RL checkpoint.

## Live verification on 2026-11-01

Job 204084 started on gpu21 (A100 SXM 80GB). At 18:45 Asia/Shanghai it had
completed 123/313 updates, saved checkpoints 1/50/100 and passed the actual
generated-token/completion-mask alignment checks. Peak allocated memory was
51.51GiB. The first rollout verified 2039 boxes, 614 extras and 250 actual
stop-token penalties. The frozen val256 evaluation ran at update 100:

| Diagnostic | Value |
| --- | ---: |
| Weighted scalar reward | 0.61937192 |
| Set reward | 0.62074330 |
| Strict reward | 0.61617200 |
| Rule-based missing-GT fraction (mean of batch fractions) | 0.30735361 |
| Extra-prediction fraction (mean of batch fractions) | 0.20518861 |
| Strict-valid response fraction | 0.98046875 |

These are reward diagnostics, not COCO AP, and there is no same-val SFT
reward comparison yet. First20/last20 training reward means were 0.61683
and 0.61937 on different training images, so they do not establish improvement.

Full 5000-image COCO val2017 SFT evaluations are now complete:

| Model | AP | AP50 |
| --- | ---: | ---: |
| Untrained Qwen3-VL-2B | 18.7387 | 25.8151 |
| JSON LoRA | 31.2397 | 46.4956 |
| Added-coordinate-token LoRA | 30.0966 | 45.5307 |
| Original-Rex-ID LoRA (RL starting point) | 29.5431 | 45.0234 |

Recovery job 204091 failed on optimizer loading due to the installed
torch2.5.1/Transformers safety restriction. Fresh global-reward control
205664 starts from the exact same SFT adapter, batch32 and fixed data order
as credit training; it is on gpu19 and has evaluator 205666. Failed recovery
outputs and all previous checkpoints are preserved. Credit evaluator 204088
remains dependent on 204084.

## Queue time and intermediate AP (2026-11-01)

Credit job 204084 was submitted at 2026-10-31 22:00:34 Asia/Shanghai and
started at 2026-11-01 15:39:42: **17h39m08s waiting**, before GPU training.
The 7–8-hour runtime estimate excludes queue time.

The existing training validation at 100/200/300 measures scalar reward and
object diagnostics, not AP. Intermediate COCO AP is now added on exactly the
same fixed val256 IDs: SFT job 205678 and credit checkpoint100 job 205679.
Future AP jobs are connected to complete checkpoint files through CPU-only
readiness gates:

| Model / step | CPU gate | AP evaluation |
| --- | ---: | ---: |
| Credit 200 | 205680 | 205681 |
| Credit 300 | 205693 (after 205680) | 205683 |
| Global 100 | 205684 | 205685 |
| Global 200 | 205694 (after 205684) | 205687 |
| Global 300 | 205688 (after 205694) | 205689 |

Results are written under `evaluation/progress_val256/` and are subset AP,
not full COCO AP. Full evaluators 204088/205666 remain attached to successful
training completion. Initial probe jobs 205671/205672 failed before inference
because the metadata view omitted the SFT's `chat_template.jinja`; copying
this file fixes both progress probes and the final RL metadata path. Those
failed attempts produced no AP results. CPU submitter 205673 could not query
sacct from its computing node and was replaced by native Slurm dependencies
on file-readiness gates. Training and saved weights were not changed.

The first complete 256-response training rollout of the fresh global control
and credit run was compared directly: **256/256 completion strings were
identical**, with first weighted reward 0.56405634 in both runs. Their initial
data/model/sampling configuration is therefore empirically aligned before the
first credit-modified optimizer update.
