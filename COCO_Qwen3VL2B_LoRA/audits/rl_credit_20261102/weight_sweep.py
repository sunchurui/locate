"""Offline coefficient calibration on saved rollouts; no training or AP claim."""
import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
args = p.parse_args()
root = Path(args.root)
weights = [.1, .25, .5, 1., 2.]
counts = dict(responses=0, tokens=0, modified_tokens=0, extras=0, stops=0)
sums = dict(base_abs=0., raw_abs=0., base_sq=0., raw_sq=0.,
            seq_base_sq=0., seq_raw_sq=0.)
extra_positive = [0] * len(weights)
stop_positive = [0] * len(weights)
flips = [0] * len(weights)
for line in (root/'rex_id_grpo_outputs_object_credit_v1/object_credit_audit.jsonl').open():
    record = json.loads(line)
    a, base = record['annotation'], record['base_advantage']
    n = len(a['response_ids'])
    raw = [0.] * n
    for pred in a['predictions']:
        for idx in pred['token_indices']:
            raw[idx] += pred['credit']
        if pred['gt_index'] is None and not pred['ignored']:
            counts['extras'] += 1
            for k, w in enumerate(weights):
                extra_positive[k] += base + w * pred['credit'] > 0
    for idx, value, _ in a['stops']:
        raw[idx] += value
        counts['stops'] += 1
        for k, w in enumerate(weights):
            stop_positive[k] += base + w * value > 0
    changed = [v for v in raw if v != 0]
    counts['responses'] += 1
    counts['tokens'] += n
    counts['modified_tokens'] += len(changed)
    raw_sq = sum(v*v for v in changed)
    sums['base_abs'] += abs(base)*n
    sums['raw_abs'] += sum(abs(v) for v in changed)
    sums['base_sq'] += base*base*n
    sums['raw_sq'] += raw_sq
    sums['seq_base_sq'] += base*base
    sums['seq_raw_sq'] += raw_sq/n
    for k, w in enumerate(weights):
        flips[k] += sum(base*(base+w*v) < 0 for v in changed)
rows = []
for k, w in enumerate(weights):
    rows.append(dict(box_weight=w, stop_weight=w,
        correction_to_base_L1=w*sums['raw_abs']/sums['base_abs'],
        correction_to_base_L2=w*math.sqrt(sums['raw_sq']/sums['base_sq']),
        response_mean_normalized_RMS_ratio=w*math.sqrt(sums['seq_raw_sq']/sums['seq_base_sq']),
        sign_flip_fraction_of_modified=flips[k]/counts['modified_tokens'],
        extra_boxes_still_positive_fraction=extra_positive[k]/counts['extras'],
        penalized_stops_still_positive_fraction=stop_positive[k]/counts['stops']))
report = dict(scope='Valid responses from the existing credit run, holding samples and base advantages fixed. '
                    'Advantage coefficient statistics, not parameter gradients or predicted AP.',
              counts=counts, rows=rows)
Path(__file__).with_name('weight_sweep_report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
