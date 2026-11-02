# COCO experiment requirements

Read `PROJECT_MEMORY.md` before editing training, validation, inference or reports.

- All evaluations must ignore detection scores and generated order, and count
  every parsed output box. This applies to baseline, SFT, RL, progress validation
  and every inference ablation. Extra/duplicate boxes must count as false positives.
- Never fabricate descending scores from output order, use model/voting scores
  to rank evaluation boxes, or treat constant-score COCOeval as order-independent.
- Use same-image/class one-to-one set matching, maximize valid match count before
  IoU, and document crowd handling. No evaluation-stage top-k, NMS or score cutoff.
- Algorithm-internal candidate selection is distinct from evaluator ranking.
  Preserve each method's actual output; do not silently tune/prune it using GT.
- The user requires modified, score-free AP results. State and persist the exact
  definition. P/R/F1 are supplemental; do not silently substitute F1 for AP or
  label a custom no-ranking AP as standard COCO AP.
- Keep old ranked-AP artifacts for provenance, mark them superseded, and write
  new results separately. Recompute from saved predictions whenever available.
- Verify permutation and score invariance, duplicate penalties, maximum-cardinality
  matching and empty/crowd behavior before publishing changed metric results.
- Training requires GPU memory >=80GB. Do not delete data/models/checkpoints.
