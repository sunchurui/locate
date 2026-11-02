# Paired RL from SFT epoch3

Both policy and frozen reference load exactly
`rex_id_sft_outputs_5epoch_b64/checkpoint-5547`, Qwen3-VL-2B Rex-ID LoRA.
The existing RL overlay contains the original model shards and tokenizer
metadata, not merged SFT weights. Tokenizer/processor files are checked against
the selected checkpoint before execution. Each run stores source snapshots
and the common base/data hashes in `paired_experiment.json`.

Submit `train.sbatch` with `RL_VARIANT=global` or `credit_norm`.
Both reuse the existing Swift GRPO loop, fixed train10000/val256, seed20260812,
313 updates, 32 image prompts x8 answers, microbatch32 x accumulation8,
learning rate5e-7, original scalar rewards and frozen SFT reference.
The optional v2 object-credit integration is the sole method difference:
per-response population z-score, box/stop weights0.1, independent stop penalty.
Output directories are separate and never overwrite prior runs.

Checkpoints every50 updates; reward validation every100 updates. At100/200/300,
a callback also runs actual val256 generation and Rex official P/R/F1 on the
same allocated >=80GB GPU. Failed/timed-out progress evaluations are recorded
in evaluation_status.json without discarding RL progress. No CPU checkpoint
readiness task occupies a concurrent-job slot while waiting for future files.
Full5000-image evaluators run separately after successful training. A separate
full evaluation of the epoch3 SFT base supplies the common comparison.

On 2026-11-02 the user explicitly selected the latest PROJECT_MEMORY protocol:
reuse Rex official P/R/F1, not the withdrawn AP_set or generation-order AP.
All80-class-query outputs are retained as the all-query evaluation input.
Original FastEval internals (constant confidence, same-score handling and
maxDets) and the positive-only reference are documented in official_metrics.py;
we do not claim permutation invariance or call this official COCO AP.
