# Current metric: global endpoint micro P/R/F1

User authorization2026-11-02: replace ALL primary results with audited variant E
from audits/REX_F1_ROOT_CAUSE_20261102.md. This explicitly supersedes earlier
exact-author-reducer requirements.

## Definition

At each IoU t=.50,.55,...,.95, retain original Rex native per-image/category
matching, then sum TP_t, FP_t, FN_t across ALL images AND categories:

- P_t=TP_t/(TP_t+FP_t)
- R_t=TP_t/(TP_t+FN_t)
- F1_t=2TP_t/(2TP_t+FP_t+FN_t)

Zero denominators return0. Zero-performance classes remain in the counts;
predictions of absent-GT categories contribute FP under native ignore rules.
Report IoU50 and IoU95. meanIoU: average ten P_t and ten R_t separately, then
compute their harmonic mean. This is exactly audited variant E, NOT mean(F1_t).
Per-IoU counts and metrics are saved for verification.

Project protocol rex_matching_global_endpoint_micro_v1 is NOT original Rex F1,
GroundingPI GAM, COCO AP or AP_set. Small subsets may still legitimately differ
from full data. Scoring is never tuned to force scores to agree.

## Retained matching and changed aggregation

Base Rex commit6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e. Vendored original source
and hash lock remain unchanged. official_eval/build_counts_engine.py creates
an isolated build, exports existing cumulative TP/FP/GT BEFORE PR interpolation,
includes absent-GT-category FP, and initializes unused legacy precision variables.
Matching remains unchanged; full5000/fixed256 per-category native counts are
compared against the audit arrays, and disjoint-image additivity is checked.

Retain native IoU/crowd/ignore/maxDets100 per image/category. Every detection has
constant0.2; generated position, token probability and vote counts do not supply
scores for evaluation. Native within-image equal-score handling still exists;
endpoint aggregation does not rank detections across images or use PR curves.
Do not claim native greedy matching is invariant to within-image box order.
All parsed outputs from our all80-category task are retained. No GT-positive
category filtering in the primary metric.

## Source of previous issues

Pinned original source has stale/uninitialized precision extraction variables.
Its safe_mean drops zeros, and precision comes from an interpolated PR curve.
Category macro averaging is a valid weighting choice, but differs from pooled
counts. Our external converter was checked against unchanged original converter
and CLI: identical converted input yields elementwise identical P/R arrays.
See the root-cause audit for the distinction between bugs and metric definitions.

## Run and outputs

```bash
venv/bin/python official_eval/build_counts_engine.py  # once per machine
venv/bin/python audits/verify_micro_migration.py      # needs saved baseline/audit
venv/bin/python rescore_micro.py --root "$PWD"
```

Canonical micro_metrics.py/rescore_micro.py. official_metrics.py/unranked_metrics.py
are compatibility APIs for current jobs, not author aggregation. Original author
reproduction remains in legacy_rex_metrics.py. rescore_official.py and
rescore_unranked.py delegate to the current metric.

Current micro_metrics.json, set_metrics.json compatibility copy, and
evaluation/micro_summary.json/.md. All results carry protocol, image coverage,
source annotation hash and native counts. Previous rex_official_metrics and
other old metrics are historical only; never mix them into current curves or
deltas. Predictions, models, data, checkpoints and old source records remain.
