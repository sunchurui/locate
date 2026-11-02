# Author-aligned evaluation: Rex-Omni P/R/F1

The user requested the existing works' protocol, not an invented no-ranking AP.
The earlier AP_set=P*R protocol is withdrawn. Standard ranked AP and the custom
AP_set are not current primary metrics.

## Exact reference

Rex commit6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e:

- evaluation/fastevaluate: compile and call the unmodified native evaluator.
- evaluation/metrics/coco_lvis_metric.py: reuse safe_mean and f1 directly.
- evaluation/utils/convert_coco_lvis_to_standard_format.py: fixed confidence0.2.
- evaluate(gt,pred,0,0,"coco"), same native maxDets/crowd/tie behavior.
- Report P/R/F1 at IoU.50,.95 and averaged-IoU metrics from the author's script.
  In particular safe_mean excludes entries<=0; the mean-IoU F1 is computed from
  the returned averaged P and R, not by our own reformulation.

The underlying native evaluator contains sorting of fixed-score entries and
precision-curve code. It is incorrect to claim it has no sorting at all. No
confidence derived from generated position, model tokens or voting is supplied.
The primary paper metrics are P/R/F1, not that engine's AP output.

## Prompt/filtering scope

The paper queries GT categories; official eval_coco.sh runs --positive_only.
Our frozen experiment queries all80 classes. Primary replay keeps every parsed
output and uses the original metric core. We separately report the source
script's GT-positive-only filtering reference. This is not a claim to have
regenerated answers under the paper's GT-category prompt.

GroundingPI's paper also reports P/R/F1. Its current GAM scorer has distinct
per-image averaging and exact-box deduplication rules. Its source is preserved
for inspection, but Rex results are not labeled GroundingPI results.

## Reproduction

Build the exact original code in an isolated directory, without altering the
shared training environment:

```bash
cd official_eval/upstream/rex/evaluation/fastevaluate
/path/to/venv/bin/python setup.py build_ext --inplace
cd /path/to/COCO_Qwen3VL2B_LoRA
venv/bin/python rescore_official.py --root "$PWD"
```

New files: rex_official_metrics.json and evaluation/rex_official_summary.*.
set_metrics.json is only a compatibility copy of the same current author result.
Previous custom values are preserved as legacy_custom_set_metrics.json where
available. No model/checkpoint/data is removed and no inference is redone.

The original source is vendored under official_eval/upstream with its license.
Each evaluation runs in a short-lived CPU subprocess to isolate the author's
native allocations; the original code and reducers are not modified.
