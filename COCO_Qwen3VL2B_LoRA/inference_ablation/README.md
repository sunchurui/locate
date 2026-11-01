# Frozen Rex-ID SFT: inference-only ablations

All runs load `model/` plus **`rex_id_sft_outputs_a800_80`**, not an RL
checkpoint. All parameters are frozen. M41 is excluded at the user's request.
The experiment compares M06/M44/M45/M46 against a paired greedy baseline.

## Fixed protocol

- Use exactly the existing `rl_val_manifest.json`: 256 COCO val2017 images,
  disjoint from training. Validate annotation SHA256 against the manifest.
- Seed 20261101; deterministically hash each image/request/chunk into its seed.
  Save actual batch sizes and seeds, including after automatic OOM reduction.
- Preserve Rex original IDs 150643–151642 and q/999 geometry. Use the SFT's
  processor and chat template. Generation limit is 1024 tokens.
- Every method generates and retains its own unchanged greedy baseline using
  the same frozen model, hardware, parser and evaluator. Methods receive image
  metadata and the fixed category vocabulary, never per-image ground truth.
- Report COCO AP/AP50/AP75/AR and additional sampled answers, generated tokens,
  runtime and peak memory. The official greedy generation-order ranking stays
  the baseline scoring convention; ranking changes are separately labeled.
- Primary comparisons are declared in `config.json` before looking at results.
  An exploratory effect is delta AP >=1 percentage point with a paired
  image-bootstrap 95% interval above zero (200 draws). This is screening on a
  small validation subset, not proof of general improvement. Confirm promising
  methods on new images/full validation without tuning on that test set.
- The first four real images per shard are a pipeline pilot, not efficacy
  evidence. M45 has four disjoint image shards, whose union is the same256.

## Methods and exact source reuse

The source URLs, commits and SHA256 values are in `sources.lock.json`.
`setup_sources.py` downloads those exact files. No upstream trainer or
model-loading script is executed. Relevant pure functions/AST fragments are
loaded by `upstream.py`; raw upstream files are retained locally for comparison.

### M06 — CANL coordinate confidence

Source: [CANL](https://github.com/ZJU-REAL/CANL), commit
`a928391d2a82f7f5fc6f66fe55e5cdd52845e3a0`.
The author's trainer applies softmax to generated processed logits, sums the
chosen coordinate-digit probabilities and divides by their count. It samples
with T=1, top_k=50, top_p=1 in the inspected generation path.

Our Qwen3/Rex port recognizes the 1000 coordinate tokens rather than the
author's digit IDs15–24, and averages over all boxes in the answer rather than
one GUI point. It generates16 candidates and records nested budgets1/4/8/16.
Selection never uses GT. Primary: `best16_order`, highest mean coordinate
confidence, keeping the baseline generation-order detection scores. Additional
variants separate candidate selection from per-box confidence ranking:
`greedy_confidence`, `sample0_order`, and `bestK_confidence`.

The streaming recorder is appended after the installed HF generation
processors and returns logits unchanged. It stores only the 1000-bin
probabilities needed for the selected tokens, not full-vocabulary histories.
On the first actual image it checks these probabilities against
`softmax(generate_output.scores)` and fails on disagreement.

### M44 — user-approved per-box zoom adaptation

Source: [zoom-consistency-routing](https://github.com/omxyz/zoom-consistency-routing),
commit `4ddfd141585e3cbfacb5a280f42df3bcbe3dc73b`.
Directly reuse `compute_crop_box`, `zoom_consistency` and `remap` from `src/zoom.py`.

The paper's application routes between two different model pipelines. This
experiment instead uses the same frozen LoRA for initial detection and zoom
verification, as explicitly approved by the user. For each initial box, crop
half the image width/height around its center, resize to the original image
size, and ask for that class nearest the crop center using Rex markers.
Map candidate corners back through the author's geometry, associate by maximum
same-class IoU >=0.1 to avoid jumping to an unrelated object, and replace only
when crop-normalized center distance <=100. These two thresholds are our fixed
adaptation conventions, not claimed paper defaults. Report `zoom_always` as a
control for zoom without the consistency gate. Primary: `zoom_consistency`.
Output ordering/scores are retained, so the primary change is box geometry.

### M45 — original spatial voting, separately for objects

Source: [GUI-RC](https://github.com/zju-real/gui-rcpo), commit
`af15ed5fe8b89b0fe5043a3e94f2984c7b126a4b`, `GUI-RC/evaluation.py`.
The author's max-vote connected-region/largest-area selection block is executed
directly, with only its final GT accuracy check removed. A runtime AST check
rejects any GT reference remaining in the inference fragment.

Generate64 answers per image. Code temperature0.7 and paper temperature0.5 are
both retained as separate variants; top_p0.95, top_k0, max1024. For multi-object
COCO output, associate same-class boxes across draws by one-to-one maximum IoU
with threshold0.5. Each cluster gets at most one vote from each draw. Matching
uses predictions only. Different objects are not collapsed into one global
maximum; association is the explicit multi-object extension to the GUI setup.

Primary `t0.7_vote` returns each cluster's original maximum-vote consensus
region, scoring it by peak votes/64. The original paper evaluates click/center
accuracy, so the consensus region need not preserve a full object's extent.
`tT_member` separately chooses a sampled box whose center is nearest the vote
region, retaining its extent; this is a box-task adaptation, not the paper's
native output. `tT_sample0` isolates the effect of voting from sampling alone.
All memberships, vote counts, regions and chosen members are saved.

### M46 — OpenRef MCC

Source: [OpenRef-Benchmark](https://github.com/zongjianwu/OpenRef-Benchmark),
commit `5c57708189aa45c51ea79b03a8426190c33ae3da`,
`mcc_qwen3_2B_grefcoco.py`. Reuse the author's actual counting and recheck
prompt expressions through AST extraction, first-integer parsing/default1,
count-conflict trigger and nonempty-refinement fallback.

Adapt the referring query to "objects belonging to the 80 MS COCO categories"
and append the existing Rex output-format convention to the checker prompt.
The counter is still this same frozen SFT LoRA, not a stronger external model.
It requests a single total count, with a 10-token generation budget. The public
checker uses 300 new tokens for referring-expression localization; our checker
keeps the common 1024-token COCO multi-object budget. This is an explicit task
adaptation, and truncated outputs are retained. If count
differs from parsed box number, recheck; if no boxes are parsed, keep the initial
answer, as in the public implementation. Strict integer-response validity and
raw count/recheck outputs are recorded to expose possible prompt/task mismatch.
Primary: `mcc`.

## Run and inspect

The existing Qwen3-VL SFT environment supplies torch/transformers/peft,
qwen-vl-utils, Pillow, SciPy and pycocotools. Do not install upstream training
environments into the active experiment environment.

```bash
cd /path/to/COCO_Qwen3VL2B_LoRA
venv/bin/python inference_ablation/setup_sources.py
sbatch --export=ALL,METHOD=m06 inference_ablation/evaluate.sbatch
sbatch --export=ALL,METHOD=m44 inference_ablation/evaluate.sbatch
sbatch --export=ALL,METHOD=m46 inference_ablation/evaluate.sbatch
sbatch --array=0-3 --export=ALL,METHOD=m45,NUM_SHARDS=4 inference_ablation/evaluate.sbatch
# Run aggregation after all jobs, preferably on CPU:
sbatch --dependency=afterany:JOB_M06:JOB_M44:JOB_M45:JOB_M46 inference_ablation/analyze.sbatch
```

GPU evaluation uses available Ampere/newer types with BF16. The default is
A6000 48GB; H100 is also supported. Each comparison is paired on its own device,
which is recorded.
In particular, cross-device tiny numerical differences are not attributed to
a method. Overrides use standard sbatch `--gres`, `--mem`, `--cpus-per-task`.
V100 FP16 and FP32 pilots both repeated coordinates on real image463522 to the
1024-token limit, whereas the established BF16 baseline ended normally after
109 tokens. Both pilots are preserved separately and excluded from method
comparison. The launcher rejects pre-Ampere hardware to retain the established
BF16 evaluation convention. Generation batches shrink on OOM without changing the
number of sampled answers; actual batches and seeds are recorded.

Each `results/METHOD/shardN/` contains immutable experiment config, image IDs,
adapter hash, environment, a source-code snapshot per Slurm job, raw generation
IDs/probabilities/prompts, atomic per-image records, pilot/full metrics and cost
summary. Cached generations allow interrupted jobs to continue in the same
output with the same configuration; changing the config requires a new output.
`analyze.py` verifies complete, nonoverlapping image coverage and aggregates
all variants. Image-bootstrap draws use new IDs for repeated images rather than
letting COCOeval silently deduplicate the draw. No model, data or checkpoint is
deleted or trained by this package.
