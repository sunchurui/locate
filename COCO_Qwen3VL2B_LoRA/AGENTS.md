# Mandatory COCO evaluation requirements

Read PROJECT_MEMORY.md and EVALUATION_PROTOCOL.md before changing evaluation.

- Follow the actual Rex-Omni / GroundingPI author evaluation; do not invent metrics.
- Withdrawn: generated-order confidence AP, voting-confidence AP, custom P*R AP_set.
- Rex detection metrics are author P/R/F1, not AP. Reuse unmodified FastEval C++
  and official safe_mean/f1 reducers at the locked commit. Generated boxes use
  the author's constant0.2, never artificial descending or learned scores.
- Do not claim that the original engine contains no internal sorting, tie handling
  or maxDets. Do not silently substitute custom Hungarian or micro F1.
- Keep all outputs for our existing all80-class task. Report the official
  GT-positive-only filtering reference separately; never silently remove false
  categories using GT or claim our prompt is the original paper prompt.
- GroundingPI GAM and Rex FastEval differ in matching/averaging/deduplication.
  State the exact source, version and task route; do not interchange their names.
- Use official_metrics.py / rescore_official.py. Old APIs are compatibility shims.
  Current results are rex_official_metrics.json and rex_official_summary.*.
- Preserve obsolete results for provenance, mark them withdrawn from primary
  reporting, and recompute saved predictions rather than regenerating silently.
- Training GPU memory>=80GB. Never delete model/data/checkpoints.

Known upstream metric issue: read audits/REX_F1_ROOT_CAUSE_20261102.md. Precision carry-over, zero filtering and interpolated PR make small-subset scores optimistic. Diagnostics are separate; do not silently change the selected author protocol.
