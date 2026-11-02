"""Recompute all saved COCO detections with the user's score-free protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from unranked_metrics import evaluate, load_predictions, save_metrics, self_check


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root',required=True)
    p.add_argument('--predictions'); p.add_argument('--image-ids'); p.add_argument('--output')
    args = p.parse_args();root = Path(args.root)
    self_check()
    annotation = root/'data/coco2017/annotations/instances_val2017.json'
    dataset = json.loads(annotation.read_text())
    manifest = json.loads((root/'rl_val_manifest.json').read_text())
    all_ids = {i['id'] for i in dataset['images']};fixed = set(manifest['image_ids'])
    registry = []
    def record(name,rows,output,expected=None):
        result = evaluate(dataset,rows)
        result['source'] = name; result['annotation_sha256'] = hashlib.sha256(annotation.read_bytes()).hexdigest()
        result['image_ids'] = sorted(rows)
        result['coverage'] = ('complete' if set(rows)==set(expected) else 'partial') if expected else 'saved_prediction_coverage'
        result['expected_images'] = len(expected) if expected else None
        path = save_metrics(output,result)
        row = {'source':name,'images':len(rows),'coverage':result['coverage'],'metrics':str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
               'iou50':result['thresholds']['0.50']['micro'],'iou75':result['thresholds']['0.75']['micro'],
               'AP_set_50_95':result['AP_set_50_95'],'AP_set_50':result['AP_set_50'],'AP_set_75':result['AP_set_75'],
               'mean_F1_50_95':result['mean_micro_F1_50_95']}
        registry.append(row);print('UNRANKED_RESULT',json.dumps(row),flush=True)
        return result
    if args.predictions:
        rows = load_predictions(args.predictions)
        expected = all_ids
        if args.image_ids:
            value = json.loads(Path(args.image_ids).read_text());expected = value.get('image_ids') if isinstance(value,dict) else value
            if not set(expected).issubset(rows):raise ValueError('Incomplete prediction coverage')
            rows = {i:rows[i] for i in expected}
        record(str(args.predictions),rows,Path(args.output or Path(args.predictions).parent),expected)
        return
    paths = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ['venv','model','data','.git','runtime_tmp','upstream_sources','source','code_snapshots','hardware_pilots','sft_ablation_smoke']
                   and (not d.startswith('checkpoint-') or 'evaluation' in Path(base).relative_to(root).parts)]
        for name in files:
            if name=='predictions.jsonl':paths.append(Path(base)/name)
    for path in sorted(paths):
        if path.stat().st_size==0:continue
        rows = load_predictions(path)
        if not rows:continue
        relative = str(path.relative_to(root))
        if not set(rows).issubset(all_ids):raise ValueError(f'Non-val images in {path}')
        expected = fixed if ('val256' in relative or 'progress_eval' in relative or 'progress_val256' in relative) else all_ids
        record(relative,rows,path.parent,expected)
        if relative=='evaluation/rex_id_val2017_a800_80/predictions.jsonl':
            record(relative+'#fixed256',{i:rows[i] for i in fixed},path.parent/'unranked_fixed256',fixed)
    for method in ['m06','m44','m45','m46']:
        rows=[]
        for path in sorted((root/'inference_ablation/results'/method).glob('shard*/records/*.json')):
            rows.append(json.loads(path.read_text()))
        if not rows:continue
        ids = [r['image_id'] for r in rows]
        if len(ids)!=len(set(ids)):raise ValueError('Overlapping inference shards')
        target = root/'inference_ablation/results'/method/'unranked'
        variants = ['baseline',*sorted(rows[0]['variants'])]
        baseline = None;summary={}
        for variant in variants:
            mapping = {r['image_id']:r['baseline'] if variant=='baseline' else r['variants'][variant] for r in rows}
            result = record(f'inference/{method}/{variant}',mapping,target/variant,fixed)
            if variant=='baseline':baseline=result
            summary[variant] = {'images':len(mapping),'coverage':result['coverage'],
                                'AP_set_50_95':result['AP_set_50_95'], 'AP_set_50':result['AP_set_50'], 'AP_set_75':result['AP_set_75'],
                                'delta_AP_set_points':100*(result['AP_set_50_95']-baseline['AP_set_50_95']),
                                'iou50':result['thresholds']['0.50']['micro'],
                                'delta_F1_50_points':100*(result['thresholds']['0.50']['micro']['F1']-baseline['thresholds']['0.50']['micro']['F1'])}
        (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=root/'evaluation/unranked_summary.json'
    report.write_text(json.dumps({'protocol':'unranked_detection_set_v1','results':registry},indent=2)+'\n')
    lines=['# Unranked evaluation results','',
           'All predictions retained. Same-image/class one-to-one matching; no scores/top-k/NMS. Values are percentages. Partial coverage is explicit.','',
           'AP_set is custom single-point P*R per class/IoU, averaged across classes and IoU=.50:.05:.95; not standard ranked COCO AP.','',
           '| Source | Images | Coverage | AP_set | AP_set50 | AP_set75 | P@.5 | R@.5 | F1@.5 |',
           '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in registry:
        v=r['iou50'];lines.append(f'| {r["source"]} | {r["images"]} | {r["coverage"]} | {100*r["AP_set_50_95"]:.2f} | {100*r["AP_set_50"]:.2f} | {100*r["AP_set_75"]:.2f} | {100*v["precision"]:.2f} | {100*v["recall"]:.2f} | {100*v["F1"]:.2f} |')
    report.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('ALL_UNRANKED_RESCORING_COMPLETE',len(registry),flush=True)


if __name__=='__main__':main()
