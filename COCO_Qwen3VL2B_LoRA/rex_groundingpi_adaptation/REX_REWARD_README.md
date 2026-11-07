# Rex marker output: GroundingPI-inspired reward adapter

This adds a separate reward entrypoint to the existing Rex-Omni / GroundingPI-inspired adaptation. It does not modify the JSON baseline. The parser is shared with the opt-in object-credit module to avoid giving local credit to outputs that fail reward-format parsing.

* Reward entrypoint: `references/Rex-Omni/finetuning/verl/configs/rex_groundingpi_reward.py:compute_score`
* Shared parser: `references/Rex-Omni/finetuning/verl/configs/rex_format.py`
* Output uses `<|object_ref_start|>category<|object_ref_end|><|box_start|><x1><y1><x2><y2>,...<|box_end|>`, optionally multiple categories, category payload `None`, or a top-level `None`. An optional trailing `<|im_end|>` is accepted.
* The Rex-token SFT serializes 0..999 bins with `int(999*x/width)`; this adapter scales prediction bins by 1000/999 to match the existing GroundingPI plugin's 0..1000 GT coordinates. This differs deliberately from official Rex-Omni's 1000 divisor and should only be used with the Qwen3 Rex-token SFT protocol.
* Matching/score formulas remain the existing `_set_reward` and `_strict_reward` from `groundingpi_reward.py`, and `overall = 0.7*set + 0.3*strict` remains the log score. The existing trainer patch separately normalizes the two reward components to compute advantages; it does not simply normalize `overall`.
* `rex_format_valid` is recorded as a metric. Malformed output, invalid or incomplete boxes, trailing punctuation and reversed/degenerate boxes receive zero component rewards and no local credit. No coordinate repair is done. Ordering, duplicate, format and other strict-score conventions remain the explicit GroundingPI-inspired approximations in the parent README, including the ordering bonus; they are not an official GroundingPI implementation or the official Rex-Omni reward formula.

For a future integrated launch, the relevant override values are:

```
worker.reward.reward_function=/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA/references/Rex-Omni/finetuning/verl/configs/rex_groundingpi_reward.py:compute_score
worker.reward.skip_special_tokens=false
algorithm.groundingpi_rewards=true
algorithm.use_kl_loss=true
algorithm.object_credit_enabled=true
algorithm.object_credit_weight=0.1
algorithm.object_credit_allowed_labels=coco80
```

Unmatched-box and missed-object penalties also require `algorithm.object_credit_complete_gt=true`. COCO annotations exclude crowd instances and can be incomplete; turning this on is an experimental assumption and requires dataset review. The example is a configuration fragment, not a complete launch command or evidence of runnability. The reward manager fails early if this plugin is used with `skip_special_tokens=true`.

The pure CPU `check_rex_reward.py` validates examples and compares scores to the existing JSON plugin with identical normalized boxes; it stubs the underlying IoU helper and does not exercise the actual training runtime, tokenizer, model or GPU. Python syntax checks have also passed. Existing unresolved model/LoRA/reference, dataset loader and validation integration blockers remain; see `OBJECT_CREDIT_README.md`.
