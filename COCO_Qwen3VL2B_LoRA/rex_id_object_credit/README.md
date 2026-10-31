# Object-aligned credit with termination penalties

This is an optional extension of the **working ms-swift 4.6.0 GRPO path** in
`../rex_id_grpo`. It does not use the older, incomplete veRL object-credit
prototype. Scalar set/strict rewards, their GDPO normalization, the upstream
clipped GRPO loss, frozen SFT reference and optimizer are preserved.

The policy starts from the original-ID **Qwen3-VL-2B SFT LoRA**, just like the
global-reward baseline. It does not start from that baseline's RL checkpoint.
Coordinate IDs are 150643–151642 (`<0>`–`<999>`); geometry uses q/999.

## Credit definition

Generate 8 complete responses for each of 32 distinct images, then compute
credit before splitting the 256 responses into training microbatches.

1. In each response, match same-class predictions to GT with a maximum-total-IoU
   one-to-one assignment. Strictly positive IoU permits association; dummy
   columns allow predictions to remain unmatched. There is no 0.5 localization
   reward threshold. Prediction/GT order is fixed; identical-box ties retain the
   earlier generated box. General ties use deterministic SciPy assignment.
2. For GT j in response k, quality q[k,j] is matched IoU, or zero when missing.
   Invalid-format responses also have zero GT quality and receive no added
   token credit. For a matched box, C[k,j] = q[k,j] - mean_k(q[k,j]) over **all 8**
   responses for this same image and GT. There is no standard-deviation scaling.
3. An unmatched evaluable box receives C=-1 once. An unmatched same-class box
   with intersection/prediction-area >=0.5 in a COCO crowd/ignore region receives
   C=0. The ignore rule is an experimental convention. It does not establish
   that every unannotated real-world object is a false positive.
4. Only normally terminated, strictly valid responses receive stop penalties:
   - For a class present in the output, C_stop = -missing_class / GT_class at
     its **last actually generated `<|box_end|>`**. A class written with `None`
     still counts as present and can receive this penalty.
   - For entirely absent classes, C_EOS = -missing_absent_classes / all_GT at
     the **actually generated terminal EOS**. These absent-class misses are
     not also charged to another class's box_end.
   - Truncated responses receive no added termination credit. No synthetic EOS,
     missing box, or other nonexistent token is trained. Correct empty outputs
     have no added penalty.

For the four coordinate tokens of a box:

```
A_token = A_global + 0.1 * C_box
```

For an eligible class-ending or EOS token:

```
A_token = A_global + 0.1 * C_stop
```

Other tokens retain the original global advantage. Corrections are bounded;
negative correction does **not** guarantee negative final advantage. This is
an intentional experimental design, not a claimed GroundingPI implementation.
Class labels and individual coordinate errors do not receive separate rewards.

## Execution and reproduction

Python 3.12, existing GRPO dependencies, and SciPy 1.13.1 are used. Run:

```bash
export EXPERIMENT_ROOT=/path/to/COCO_Qwen3VL2B_LoRA
bash "$EXPERIMENT_ROOT/rex_id_object_credit/setup.sh"
python "$EXPERIMENT_ROOT/rex_id_grpo/prepare_rl.py" --root "$EXPERIMENT_ROOT"
cd "$EXPERIMENT_ROOT"
sbatch rex_id_object_credit/train.sbatch
# Replace TRAIN_JOB with the returned numeric ID:
sbatch --dependency=afterok:TRAIN_JOB \
  --export=ALL,OUTPUT_DIR="$EXPERIMENT_ROOT/rex_id_grpo_outputs_object_credit_v1" \
  rex_id_grpo/evaluate.sbatch
```

Data, prompts, subset seed 42, offline ordering/training seed 20260812 and the
313-update epoch are the same as the baseline. The wrapper creates crowd
metadata for exactly the original train10000/val256 manifests and verifies
the original annotation SHA256 values. It does not change source datasets.

Execution uses one GPU with >=80GB VRAM, microbatch32 x accumulation8 = 256
responses per optimizer update, generation memory chunks of 128, LR5e-7,
KL0.02 and clipping0.2. Batch64 exhausted memory on a longer completion in
the baseline; batch32 is used for both variants. Memory occupancy is allowed
to vary with sequence length rather than allocating unused buffers.

The 16-hour Slurm job runs independently of SSH. The first completed update
in each process is saved (including after a resume), followed by checkpoints
every 50 updates. No checkpoint retention limit is set. Validation uses the
fixed 256 images every 100 updates, reports scalar rewards and object diagnostic
metrics, and leaves validation advantages unchanged. **These are not COCO AP.**
A separate dependent job evaluates all 5000 val2017 images for COCO AP.

## Audit and limitations

`object_credit_audit.jsonl` records each valid training response's actual sampled
IDs, GT associations, quality, per-box credit, missing GT, stop positions and
global advantage. The runtime requires every credited position to match actual
sampled IDs inside the real completion loss mask, and fails on misalignment.
It checks the full 32 x 8 rollout before producing token advantages.

`audit_saved_rollouts.py` additionally audits real preexisting decoded responses.
Because those logs omit raw sampled IDs and termination reasons, retokenizing
them is only a parser/matching/coordinate-token check; actual sampling/loss-mask
alignment and EOS credit are verified in the live GPU path. Its class-ending
counts are prospective annotations assuming normal termination, not proof of
the original stop reason.

Association still has ambiguity: tiny positive overlap can count as matched,
and class-aware matching cannot separate a wrong label from a background box.
Matched but low-IoU predictions receive relative credit; their location error
is not also charged as a missed object at the stop token. Keep the unchanged
global-reward baseline for comparison.
