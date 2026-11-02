# Mandatory evaluation requirements

Read PROJECT_MEMORY.md and EVALUATION_PROTOCOL.md before evaluation changes.

- Latest explicit user decision2026-11-02: ALL primary metrics use audited variant E,
  protocol rex_matching_global_endpoint_micro_v1. Earlier exact-author-reducer
  requirements are superseded by this authorization.
- Per IoU=.50:.05:.95, pool original Rex native TP/FP/FN over images AND categories.
  meanIoU F1=harmonic(mean(P_t),mean(R_t)), not mean(F1_t). Save per-IoU counts.
- Keep zero classes and absent-GT-category FP; no PR extraction or macro averaging.
  Do not select metrics to force small/full scores equal. Use identical image IDs
  and prediction provenance when comparing methods.
- This is project micro F1, NOT original Rex F1, GroundingPI GAM, AP or AP_set.
- Native matching/crowd/ignore/maxDets100 and constant0.2 remain; no generated-order,
  model-token or voting confidence. Native within-image tie handling still exists.
- Canonical micro_metrics.py/rescore_micro.py. official_metrics.py/unranked_metrics.py
  are current compatibility APIs; original author code is legacy_rex_metrics.py.
- Vendored original sources remain hash-locked/untouched. Build isolated counts
  engine using official_eval/build_counts_engine.py. Recompute SAVED predictions.
- Current micro_metrics.json/set_metrics.json/evaluation/micro_summary.*. Mark old
  author/AP results historical and preserve provenance. Never mix old/new curves.
- Read audits/REX_F1_ROOT_CAUSE_20261102.md: original precision carry-over and zero
  dropping confirmed; identical input official CLI/wrapper parity verified.
- Mark incomplete image coverage. Training GPU>=80GB; never delete models/data/
  checkpoints or restore stopped recurring Codex monitoring.
