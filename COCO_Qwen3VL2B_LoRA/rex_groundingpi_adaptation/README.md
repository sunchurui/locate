# Rex-Omni / GroundingPI RL adaptation (incomplete)

This is a work-in-progress adaptation based on the official [Rex-Omni fine-tuning code](https://github.com/IDEA-Research/Rex-Omni/tree/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/finetuning), commit `6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e`. It is **not yet a runnable Qwen3-VL LoRA training recipe** and has not been exercised in training.

`rex_groundingpi.patch` applies to that checkout's root. It keeps Rex-Omni/Easy-R1's GRPO trainer and adds an opt-in `algorithm.groundingpi_rewards` mode. The mode reads `groundingpi_set` and `groundingpi_strict` per-response rewards, standardizes each independently across responses sharing a prompt, combines them with weights 0.7/0.3 and standardizes the resulting response batch. It requires a reference policy and KL in the loss, and fails if either reward component is missing. The default remains the original Rex-Omni behavior.

`groundingpi_reward.py` is a batch reward plugin for the upstream reward manager. It reuses `BoxIoURewardFunction.calculate_iou` from the Rex-Omni implementation and parses our Qwen JSON boxes. Its set score follows [GroundingPI section 7.6.1](https://arxiv.org/html/2609.39601): for each reference select the highest-IoU prediction first, then check its class; predictions can be reused. Empty references are outside the paper's stated definition; this adaptation assigns 1 only for an empty prediction.

The strict score uses the paper's published weights and IoU thresholds, but its detailed matching, format, ordering and penalty conventions are not all specified in the paper. This prototype uses deterministic one-to-one maximum-cardinality matching, matched IoU at threshold 0.5, exact JSON validity, count agreement `min/max`, lexicographic class/box ordering, a duplicate threshold of IoU 0.9, and an oversized threshold of 80% of image area. These are explicit approximations, not official GroundingPI code. An eventual training prompt must request the same output ordering. The plugin accepts either the fixed-subset JSON annotation string or the upstream Rex-Omni answer dictionary.

To inspect the change, clone the upstream commit, apply the patch, and place the plugin at `finetuning/verl/configs/groundingpi_reward.py`. The intended settings include `algorithm.groundingpi_rewards=true`, `algorithm.use_kl_loss=true`, `algorithm.kl_coef=0.02`, `worker.rollout.n=8`, and the plugin as `worker.reward.reward_function`. The paper uses epsilon 0.2, learning rate 5e-7 with no warmup, and a frozen SFT reference. Those settings alone do not make this prototype runnable.

Remaining integration work:

1. Add Qwen3-VL 2B model and multimodal position handling; the upstream code targets Qwen2/Qwen2.5-VL and its pinned Transformers/vLLM versions precede Qwen3-VL.
2. Load and train the existing SFT LoRA adapter, freeze vision/projector, and ensure the reference policy is the same SFT adapter. Rex-Omni's checked-in worker has no LoRA loading path.
3. Connect the 10,000-image JSONL without changing its fixed IDs; the upstream loader expects Rex-specific TSV and output tokens. Preserve the separate 256-image validation subset and wire validation: upstream currently passes `val_dataloader=None`.
4. Establish a compatible isolated environment and run reward, advantage, data, single-step GPU and resume checks before submitting a full run. The current HPC environment is missing Rex-Omni dependencies including vLLM, mmengine and torchdata.

Do not run upstream `scripts/grpo.sh` unchanged: it includes a broad `/tmp/ray/*` removal and assumes eight GPUs. No user or shared Ray files should be removed.
