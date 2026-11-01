# Fresh five-epoch Rex-ID LoRA

Starts from the original Qwen3-VL-2B model and newly initialized rank16/alpha32
LoRA, not from an existing SFT/RL adapter. Original vocabulary rows150643–151642,
Rex serialization, image-first input, full COCO train2017/val2017, BF16, SDPA,
LR1e-4/cosine/3%warmup/weight decay0.01 and seed42 are retained.

The original SFT entrypoint is frozen as `baseline_train.py`. This wrapper
changes epochs to5 and global image batch to64. On the allocated >=80GB GPU,
it probes backward passes on real long COCO answers for microbatch4/8/16/32,
chooses the largest fitting within80% memory, and fills the global batch with
gradient accumulation. Probes never step the optimizer and restore RNG states.
This reserve is conservative, not a guarantee against every future OOM.

Loss on all5000 validation images is checked every250 updates and at epoch
boundaries. Checkpoints are saved every250 updates and each epoch; none are
automatically deleted. At epoch end, a subprocess on the allocated training GPU
evaluates the fixed256-image validation set, retaining the exact Rex parser and
generation-order AP convention. Epoch metrics are saved under evaluation/.
Full5000-image evaluation is submitted with afterany dependency, so a user
cancellation still evaluates the latest complete checkpoint. Cancellation
before the first checkpoint yields an explicit error, not another model.

At batch64, one epoch is1849 optimizer updates, so five epochs are9245 updates.
The previous one-epoch experiment used batch8: changes vs that run include both
batch and training duration. Validation AP, not training loss, determines if
additional training is useful. User cancellation remains the stopping decision.

Run from the experiment root: `sbatch rex_id_multiepoch/train.sbatch`.
After its jobID is known, use `sbatch --dependency=afterany:JOBID
rex_id_multiepoch/evaluate.sbatch` for the full evaluator.
