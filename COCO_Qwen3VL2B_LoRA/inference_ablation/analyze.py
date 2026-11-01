"""Aggregate disjoint shards and compute paired image-bootstrap AP intervals."""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from metrics import bootstrap_dataset, evaluate, evaluate_records


def interval(gt, records, variant, count, seed):
    baseline = {r['image_id']: r['baseline'] for r in records}
    method = {r['image_id']: r['variants'][variant] for r in records}
    ids = np.array(sorted(baseline))
    rng = np.random.default_rng(seed)
    differences = []
    for index in range(count):
        draw = rng.choice(ids, size=len(ids), replace=True).tolist()
        boot_gt, base_dt, new_ids = bootstrap_dataset(gt, baseline, draw)
        _, method_dt, _ = bootstrap_dataset(gt, method, draw)
        base_ap = evaluate(boot_gt, base_dt, new_ids)['AP']
        method_ap = evaluate(boot_gt, method_dt, new_ids)['AP']
        differences.append(100 * (method_ap - base_ap))
        if (index + 1) % 20 == 0:
            print(f'Paired bootstrap {variant}: {index + 1}/{count}', flush=True)
    return np.quantile(differences, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--bootstrap', type=int, default=None)
    args = parser.parse_args()
    root = Path(args.root)
    folder = root / 'inference_ablation'
    config = json.loads((folder / 'config.json').read_text())
    expected = set(json.loads((root / config['manifest']).read_text())['image_ids'])
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(root / 'data/coco2017/annotations/instances_val2017.json'))
    results = {'fixed_image_count': len(expected), 'methods': {}, 'm41': 'skipped by user',
               'effect_rule': config['screening_effect_rule'], 'note': 'Subset screening; not full-val AP or confirmatory evidence.'}
    lines = ['# Frozen Rex-ID SFT inference ablations', '',
             f'Fixed validation subset: {len(expected)} images. AP is shown in percentage points.', '',
             '| Method | Variant | Baseline AP | Method AP | Delta AP | Paired 95% CI |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for method in ('m06', 'm44', 'm45', 'm46'):
        records = []
        for path in sorted((folder / 'results' / method).glob('shard*/records/*.json')):
            records.append(json.loads(path.read_text()))
        ids = [r['image_id'] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Overlapping image shards for {method}')
        if set(ids) != expected:
            results['methods'][method] = {'state': 'incomplete', 'completed_images': len(ids), 'expected_images': len(expected)}
            continue
        metrics = evaluate_records(gt, records)
        primary = config['primary_variants'][method]
        count = config['bootstrap_replicates'] if args.bootstrap is None else args.bootstrap
        ci = interval(gt, records, primary, count, config['bootstrap_seed']) if count else None
        delta = metrics[primary]['delta_AP_points']
        results['methods'][method] = {'state': 'complete', 'metrics': metrics, 'primary': primary,
                                     'primary_delta_AP_95CI': ci,
                                     'exploratory_effect': bool(ci and delta >= 1 and ci[0] > 0),
                                     'baseline_cost': {key: sum(r['baseline_cost'][key] for r in records)
                                                       for key in records[0]['baseline_cost']},
                                     'extra_cost': {key: sum(r['extra_cost'][key] for r in records)
                                                    for key in records[0]['extra_cost']}}
        for name in sorted(metrics):
            if name == 'baseline':
                continue
            ci_text = f'[{ci[0]:+.2f}, {ci[1]:+.2f}]' if name == primary and ci else '—'
            lines.append(f'| {method} | {name} | {metrics["baseline"]["AP"] * 100:.2f} | '
                         f'{metrics[name]["AP"] * 100:.2f} | {metrics[name]["delta_AP_points"]:+.2f} | {ci_text} |')
        print(method, 'primary delta AP', delta, 'CI', ci, flush=True)
    for method, item in results['methods'].items():
        if item['state'] != 'complete':
            lines += ['', f'{method}: incomplete, {item["completed_images"]}/{item["expected_images"]} images.']
    lines += ['', 'Methods use the same frozen SFT LoRA. Raw candidates, conditional coordinate probabilities,',
              'count/recheck decisions, zoom geometry, votes, timing and source versions are retained.',
              'M44 is a user-approved single-model adaptation. M45 additionally reports a sampled-box',
              'extent-preserving variant; the original paper scores GUI click accuracy, not COCO box AP.']
    (folder / 'results/summary.json').write_text(json.dumps(results, indent=2) + '\n')
    (folder / 'results/summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
