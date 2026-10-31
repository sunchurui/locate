# GroundingPI components versus official Rex-Omni reward code

Inspected only tracked files at Rex commit
`6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e`, not the untracked or modified local
GroundingPI adaptation. Official source:
https://github.com/IDEA-Research/Rex-Omni/blob/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/finetuning/verl/configs/reward_func.py

| GroundingPI term | Rex overlap | Treatment |
| --- | --- | --- |
| Format | Rex extracts object/box markers and groups coordinates by four. It skips incomplete groups, repairs inverted coordinates, and does not enforce whole-response format validity. | Call its extractor for boxes after a separate strict validator. Do not label it an official binary format reward. |
| Count agreement | No independent equality/count score in the inspected box reward. Counts appear only as precision/recall denominators. | Draft now uses exact predicted-count == GT-count, 0/1, per the user's proposal. |
| F1 at .50/.75/.95 | Rex computes a continuous-IoU soft F1, not thresholded detection F1. Matching can reuse boxes. | Keep GroundingPI's three-threshold F1; reuse only Rex IoU geometry. One-to-one assignment details remain unresolved. |
| Matched IoU | Rex provides `calculate_iou` and same-class best-overlap searches, but no standalone strict matched-IoU average. | Directly call Rex `calculate_iou`; retain the averaging definition as pending. |
| Output order | No independent ordering score in the box reward. | Draft uses binary within-class x1-order validity; sorting direction is published by GroundingPI. |
| Oversized boxes | No independent penalty in the box reward. | Draft counts each oversized box lacking same-class GT support once, multiplied by -.07; thresholds remain pending. |
| Duplicate boxes | No independent penalty in the box reward. Its reusable matching can even give high soft F1 to duplicate perfect predictions. | Draft counts each later duplicate prediction once, multiplied by -.03, not duplicate pairs or a proportion. Definition/threshold remain pending. |

Rex `compute_reward` explicitly gives 1 for empty GT plus empty prediction, 0
for empty GT plus nonempty prediction, and its formulas yield 0 for nonempty GT
plus no predictions. Our valid-empty handling agrees with this. This does not
establish that GroundingPI uses the same convention for empty GT.

Rex divides decoded tokens by 1000; this experiment's SFT/evaluation use q/999.
The adapter calls the original extractor at width=height=1000, then rescales
coordinates by 1000/999. Strict validation precedes extraction, so Rex's silent
coordinate-repair behavior is not adopted as a GroundingPI rule.

The set rewards also differ: Rex finds the best **same-class** IoU separately
for every GT and every prediction, then combines soft recall/precision.
GroundingPI finds the largest overlap for each GT **before** checking its class,
and derives both soft recall and precision from that single sum. Therefore the
official Rex `compute_reward` must not replace GroundingPI's set equation.

The draft strict composition is now:

```
clip(.10*format + .10*exact_count + .50*mean_threshold_F1
     + .25*matched_IoU + .05*ordered
     - .07*oversized_unmatched_count - .03*later_duplicate_count, 0, 1)
```

Format/count/order are binary. The two penalty terms are raw per-box counts,
following the user's proposed interpretation, not confirmed GroundingPI code.
The existing global-invalid-output gate is still pending and unchanged; the
formula above describes valid parsed outputs. No training is enabled by these
changes. Generation settings, training budget, reward matching, thresholds,
format/error handling await the prior requested review. Valid-empty terminal
behavior follows official Rex under the user's overlap-reuse instruction.
