"""Aggregate frozen inference outputs using score-free, single-point AP_set."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from metrics import bootstrap_dataset, evaluate, evaluate_records


def interval(gt, records, variant, count, seed):
    baseline = {r['image_id']:r['baseline'] for r in records}
    method = {r['image_id']:r['variants'][variant] for r in records}
    ids=np.array(sorted(baseline));rng=np.random.default_rng(seed);deltas=[]
    for index in range(count):
        draw=rng.choice(ids,size=len(ids),replace=True).tolist()
        boot,base_dt,new_ids=bootstrap_dataset(gt,baseline,draw)
        _,method_dt,_=bootstrap_dataset(gt,method,draw)
        deltas.append(100*(evaluate(boot,method_dt,new_ids)['AP_set_50_95']-evaluate(boot,base_dt,new_ids)['AP_set_50_95']))
        if (index+1)%20==0:print('AP_set paired bootstrap',index+1,count,flush=True)
    return np.quantile(deltas,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--bootstrap',type=int,default=0,help='Optional paired bootstrap of the new AP_set endpoint')
    args=p.parse_args();root=Path(args.root);folder=root/'inference_ablation'
    config=json.loads((folder/'config.json').read_text())
    expected=set(json.loads((root/config['manifest']).read_text())['image_ids'])
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(root/'data/coco2017/annotations/instances_val2017.json'))
    result={'protocol':'unranked_detection_set_v1','fixed_image_count':len(expected),'methods':{},
            'definition':'AP_set=mean over IoUs .50:.05:.95 of macro-class single-point P*R. All output boxes count. Not standard COCO AP.',
            'note':'Old ranked AP summaries are superseded; no scores/top-k/NMS used by this evaluator.'}
    lines=['# Score-free inference evaluation','',result['definition'],'',
           '| Method | Variant | Images | Coverage | AP_set | Delta AP_set | P@.5 | R@.5 | F1@.5 |',
           '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for method in ['m06','m44','m45','m46']:
        records=[json.loads(path.read_text()) for path in sorted((folder/'results'/method).glob('shard*/records/*.json'))]
        if not records:result['methods'][method]={'state':'empty'};continue
        ids=[r['image_id'] for r in records]
        if len(ids)!=len(set(ids)) or not set(ids).issubset(expected):raise ValueError('Overlapping or unexpected image IDs')
        coverage='complete' if set(ids)==expected else 'partial'
        metrics=evaluate_records(gt,records);primary=config['primary_variants'][method]
        ci=interval(gt,records,primary,args.bootstrap,config['bootstrap_seed']) if args.bootstrap and coverage=='complete' else None
        result['methods'][method]={'state':coverage,'images':len(ids),'metrics':metrics,
                                  'primary':primary,'primary_delta_AP_set_95CI':ci}
        for name,values in metrics.items():
            v=values['thresholds']['0.50']['micro']
            delta=values['AP_set_50_95']-metrics['baseline']['AP_set_50_95']
            lines.append(f'| {method} | {name} | {len(ids)} | {coverage} | {100*values["AP_set_50_95"]:.2f} | {100*delta:+.2f} | {100*v["precision"]:.2f} | {100*v["recall"]:.2f} | {100*v["F1"]:.2f} |')
        print(method,coverage,'images',len(ids),'primary AP_set',metrics[primary]['AP_set_50_95'],flush=True)
    (folder/'results/set_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (folder/'results/set_summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
