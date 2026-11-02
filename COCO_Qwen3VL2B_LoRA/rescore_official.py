"""Replay all existing evaluations through Rex-Omni's unmodified native evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from official_metrics import evaluate,load_predictions,save_metrics,self_check


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--predictions');p.add_argument('--image-ids');p.add_argument('--output')
    args=p.parse_args();root=Path(args.root)
    annotation=root/'data/coco2017/annotations/instances_val2017.json'
    dataset=json.loads(annotation.read_text());all_ids={i['id'] for i in dataset['images']}
    fixed=set(json.loads((root/'rl_val_manifest.json').read_text())['image_ids'])
    registry=[]
    def record(name,rows,target,expected):
        metrics=evaluate(dataset,rows);metrics['source']=name
        metrics['annotation_sha256']=hashlib.sha256(annotation.read_bytes()).hexdigest()
        metrics['image_ids']=sorted(rows);metrics['coverage']='complete' if set(rows)==set(expected) else 'partial'
        metrics['expected_images']=len(expected);path=save_metrics(target,metrics)
        row={k:metrics[k] for k in ['source','images','detections','coverage','meanIoU','IoU50','IoU95','official_positive_only_reference']}
        row['metrics']=str(path.relative_to(root));registry.append(row)
        print('REX_OFFICIAL_RESULT',json.dumps(row),flush=True);return metrics
    if args.predictions:
        rows=load_predictions(args.predictions);expected=all_ids
        if args.image_ids:
            value=json.loads(Path(args.image_ids).read_text());expected=value.get('image_ids') if isinstance(value,dict) else value
            if not set(expected).issubset(rows):raise ValueError('Incomplete requested coverage')
            rows={i:rows[i] for i in expected}
        record(str(args.predictions),rows,Path(args.output or Path(args.predictions).parent),expected);return
    self_check()
    paths=[]
    for base,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in ['venv','model','data','.git','runtime_tmp','upstream_sources','source','code_snapshots','hardware_pilots','sft_ablation_smoke','official_eval','references']
                 and (not d.startswith('checkpoint-') or 'evaluation' in Path(base).relative_to(root).parts)]
        if 'predictions.jsonl' in files:paths.append(Path(base)/'predictions.jsonl')
    for path in sorted(paths):
        if not path.stat().st_size:continue
        rows=load_predictions(path)
        if not rows:continue
        if not set(rows).issubset(all_ids):raise ValueError(f'Non-val images: {path}')
        name=str(path.relative_to(root))
        expected=fixed if any(x in name for x in ['val256','progress_eval','progress_val256']) else all_ids
        record(name,rows,path.parent,expected)
        if name=='evaluation/rex_id_val2017_a800_80/predictions.jsonl':
            record(name+'#fixed256',{i:rows[i] for i in fixed},path.parent/'rex_official_fixed256',fixed)
    for method in ['m06','m44','m45','m46']:
        records=[json.loads(p.read_text()) for p in sorted((root/'inference_ablation/results'/method).glob('shard*/records/*.json'))]
        if not records:continue
        ids=[r['image_id'] for r in records]
        if len(ids)!=len(set(ids)):raise ValueError('Overlapping inference shards')
        summaries={}
        for variant in ['baseline',*sorted(records[0]['variants'])]:
            mapping={r['image_id']:r['baseline'] if variant=='baseline' else r['variants'][variant] for r in records}
            metrics=record(f'inference/{method}/{variant}',mapping,root/'inference_ablation/results'/method/'rex_official'/variant,fixed)
            summaries[variant]={k:metrics[k] for k in ['images','coverage','meanIoU','IoU50','IoU95','official_positive_only_reference']}
        (root/'inference_ablation/results'/method/'rex_official/summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    target=root/'evaluation/rex_official_summary.json'
    target.write_text(json.dumps({'protocol':'rex_omni_official_fasteval_all_output_replay','results':registry},indent=2)+'\n')
    lines=['# Rex-Omni official evaluator replay','',
        'Unmodified author FastEval and safe_mean/f1. No invented AP or confidence ranking. Primary retains all outputs of our all80-class prompt; positive-only official filtering reference is separate. Values in percent.','',
        '| Source | Images | Coverage | P mIoU | R mIoU | F1 mIoU | F1@.50 | F1@.95 | Positive-only F1 mIoU |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in registry:
        v=r['meanIoU'];reference=r['official_positive_only_reference']['meanIoU']['F1']
        lines.append(f'| {r["source"]} | {r["images"]} | {r["coverage"]} | {100*v["precision"]:.2f} | {100*v["recall"]:.2f} | {100*v["F1"]:.2f} | {100*r["IoU50"]["F1"]:.2f} | {100*r["IoU95"]["F1"]:.2f} | {100*reference:.2f} |')
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('ALL_REX_OFFICIAL_RESCORING_COMPLETE',len(registry),flush=True)


if __name__=='__main__':main()
