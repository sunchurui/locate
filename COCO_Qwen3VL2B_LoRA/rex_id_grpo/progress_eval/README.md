# COCO AP during training

The training loop already validates scalar rewards and object diagnostics on
the fixed 256-image validation manifest every 100 updates. That is distinct
from COCO AP. This optional workflow computes AP on the **same fixed 256 IDs**
for the SFT adapter and saved RL checkpoints 100/200/300, without pausing or
changing an active training process. Full 5000-image evaluation remains a
separate dependency after successful completion of each training job.

`evaluate.sbatch` defaults to an independent RTX A6000 48GB (evaluation may
use any GPU), batch32 with the existing automatic OOM batch reduction, four
CPU cores, 24GB RAM and greedy generation. SFT and RL checkpoint probes use
the same evaluator and image-ID manifest. The results are subset AP and must
not be compared directly with full-val AP.

`prepare_adapter.py` links the saved adapter weights into an evaluation-only
metadata view. It never modifies training checkpoints. It requires a complete
checkpoint and validates the original 1000 coordinate-token adapter rows.

Submit a specific probe from the experiment root:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=/path/to/checkpoint-100,EVAL_OUTPUT_DIR=/path/to/probe100 \
  rex_id_grpo/progress_eval/evaluate.sbatch
```

For automatic AP evaluation at a future checkpoint, submit a small CPU-only
readiness gate and make the GPU evaluation depend on its successful exit:

```bash
export CHECKPOINT_PATH=/path/to/checkpoint-200 CHECKPOINT_STEP=200
gate=$(sbatch --parsable --export=ALL rex_id_grpo/progress_eval/wait_checkpoint.sbatch)
sbatch --dependency=afterok:$gate \
  --export=ALL,CHECKPOINT_PATH="$CHECKPOINT_PATH",EVAL_OUTPUT_DIR=/path/to/probe200 \
  rex_id_grpo/progress_eval/evaluate.sbatch
```

Repeat for steps 100/200/300 of each experiment. The gate reserves one CPU,
256MB RAM and **no GPU**, checks complete checkpoint files once per minute,
and exits when ready or fails after 15 hours. Slurm then starts the independent
GPU evaluation. If training fails before that checkpoint, the gate eventually
fails and its evaluation cannot start; cancel such remaining gates when
handling the failed training. No lock or mutex is introduced. This belongs to
the Slurm evaluation pipeline, not a Codex recurring automation or notification
monitor. Computing nodes do not need access to Slurm accounting or submission
commands; those dependencies are all submitted from the login node.

The runtime registry `progress_eval_jobs.json` in the experiment root records
actual gate/evaluation IDs. The earlier accounting-based CPU watcher failed
because sacct was unavailable from its computing node and is superseded by
the filesystem gates. Processor metadata includes `chat_template.jinja`:
leaving it out causes AutoProcessor inference to fail before generation.
