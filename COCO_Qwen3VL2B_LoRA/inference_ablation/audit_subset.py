"""Explain subset/full-val AP differences using saved predictions, without inference."""
import argparse
import contextlib
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def rows(path):
    result = {}
    for line in path.open():
        item = json.loads(line)
        if item['image_id'] in result:
            raise ValueError(f'Duplicate image in {path}')
        result[item['image_id']] = item['detections']
    return result


def score(gt, predictions, ids, categories=None):
    detections = [dict(p, image_id=i) for i in ids for p in predictions[i]]
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(detections)
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = sorted(ids)
        if categories is not None:
            evaluator.params.catIds = categories
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    precision = evaluator.eval['precision'][:, :, :, 0, -1]
    per_category = {}
    for k, category in enumerate(evaluator.params.catIds):
        values = precision[:, :, k]
        valid = values[values >= 0]
        per_category[gt.cats[category]['name']] = float(valid.mean() * 100) if len(valid) else None
    return {'AP': float(evaluator.stats[0] * 100), 'AP50': float(evaluator.stats[1] * 100),
            'AP75': float(evaluator.stats[2] * 100), 'detections': len(detections),
            'images': len(ids), 'per_category_AP': per_category}


def composition(gt, ids):
    annotations = [a for i in ids for a in gt.imgToAnns[i] if not a.get('iscrowd', 0)]
    counts = Counter(a['category_id'] for a in annotations)
    size = Counter('small' if a['area'] < 32**2 else 'medium' if a['area'] < 96**2 else 'large'
                   for a in annotations)
    return {'images': len(ids), 'noncrowd_objects': len(annotations),
            'objects_per_image': len(annotations)/len(ids), 'categories_present': len(counts),
            'missing_categories': [gt.cats[i]['name'] for i in sorted(gt.cats) if not counts[i]],
            'category_instances': {gt.cats[i]['name']: counts[i] for i in sorted(gt.cats)},
            'size_percent': {k: 100 * size[k]/len(annotations) for k in ['small','medium','large']},
            'categories_with_under_5_objects': sum(0 < counts[i] < 5 for i in gt.cats)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--random-subsets', type=int, default=50)
    args = p.parse_args(); root = Path(args.root)
    out = root/'inference_ablation/results/subset_audit_20261102'; out.mkdir(exist_ok=True)
    annotation = root/'data/coco2017/annotations/instances_val2017.json'
    manifest = json.loads((root/'rl_val_manifest.json').read_text())
    assert hashlib.sha256(annotation.read_bytes()).hexdigest() == manifest['annotation_sha256']
    with contextlib.redirect_stdout(io.StringIO()):gt = COCO(str(annotation))
    all_ids = sorted(gt.imgs); selected = manifest['image_ids']
    rank = lambda i: (hashlib.sha256(f'coco2017:val2017:seed={manifest["seed"]}:image_id={i}'.encode()).digest(), i)
    reconstructed = sorted(sorted(all_ids, key=rank)[:manifest['count']])
    assert reconstructed == selected
    full = rows(root/'evaluation/rex_id_val2017_a800_80/predictions.jsonl')
    old = rows(root/'evaluation/progress_val256/sft/predictions.jsonl')
    assert set(full) == set(all_ids) and set(old) == set(selected)
    paired, method = {}, {}
    for path in sorted((root/'inference_ablation/results/m45').glob('shard*/records/*.json')):
        row = json.loads(path.read_text()); i = row['image_id']
        if i in paired:raise ValueError('Overlapping M45 shards')
        paired[i], method[i] = row['baseline'], row['variants']['t0.5_member']
    assert set(paired) == set(method) == set(selected)
    report = {'selection_reconstructed_exactly': True, 'selection': manifest['selection'],
              'seed': manifest['seed'], 'annotation_sha256': manifest['annotation_sha256'],
              'full_composition': composition(gt, all_ids), 'subset_composition': composition(gt, selected)}
    save = lambda: (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    for name, predictions, ids in [('full_saved_predictions_full5000',full,all_ids),
            ('full_saved_predictions_fixed256',full,selected), ('old_subset_saved_predictions',old,selected),
            ('inference_paired_baseline',paired,selected), ('m45_t05_member',method,selected)]:
        report[name] = score(gt,predictions,ids);save()
        print(name,report[name]['AP'],flush=True)
    present = sorted({a['category_id'] for i in selected for a in gt.imgToAnns[i] if not a.get('iscrowd',0)})
    report['full5000_on_subset_present_categories'] = score(gt, full, all_ids, present);save()
    report['saved_prediction_comparison'] = {
        'full_vs_old_subset_exact_images': sum(full[i] == old[i] for i in selected),
        'full_vs_paired_exact_images': sum(full[i] == paired[i] for i in selected)}
    report['random_subsets'] = []
    for seed in range(1000,1000+args.random_subsets):
        key = lambda i: (hashlib.sha256(f'coco2017:val2017:seed={seed}:image_id={i}'.encode()).digest(),i)
        ids = sorted(sorted(all_ids,key=key)[:len(selected)])
        result = score(gt,full,ids)
        report['random_subsets'].append({'seed':seed,'AP':result['AP']})
        values = np.array([r['AP'] for r in report['random_subsets']])
        report['random_subset_summary'] = {'completed':len(values),'planned':args.random_subsets,
                'mean_AP':float(values.mean()),'median_AP':float(np.median(values)),
                'min_AP':float(values.min()),'max_AP':float(values.max()),
                'AP_quantiles_025_975':np.quantile(values,[.025,.975]).tolist(),
                'fraction_at_or_above_fixed256':float(np.mean(values>=report['full_saved_predictions_fixed256']['AP'])),
                'note':'Empirical variation across distinct random subsets; not a confidence interval for method gain.'}
        save();print('random subsets',len(values),report['random_subset_summary'],flush=True)
    print('AUDIT_COMPLETE',str(out/'audit.json'),flush=True)


if __name__ == '__main__':main()
