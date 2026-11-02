"""Cache-only support filtering/NMS ablation of the existing multi-object M45.

All selection uses saved predictions only; GT is loaded by the scorer afterwards.
Primary is fixed before replay: T=.7, author vote-region, support>=16/64, NMS.7.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from micro_metrics import evaluate,save_metrics,PROTOCOL


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def select(clusters,minimum,nms,box_key):
    eligible=[i for i,c in enumerate(clusters) if c['support']>=minimum]
    kept=[]
    # Support ranks candidates for POSTPROCESSING only, never for evaluation AP.
    ranked=sorted(eligible,key=lambda i:(-clusters[i]['support'],-clusters[i]['peak_votes'],i))
    for i in ranked:
        c=clusters[i]
        if nms is not None and any(c['label']==clusters[j]['label'] and iou(c[box_key],clusters[j][box_key])>=nms for j in kept):continue
        kept.append(i)
    # Preserve original output order so filtering alone does not reorder native matches.
    return [clusters[i] for i in sorted(kept)]


def detections(clusters,box_key,categories):
    result=[]
    for c in clusters:
        if c['label'] not in categories:continue
        x1,y1,x2,y2=c[box_key]
        result.append({'category_id':categories[c['label']],'bbox':[x1,y1,x2-x1,y2-y1],'score':1.})
    return result


def self_check():
    a={'label':'a','support':16,'peak_votes':16,'region':[0,0,10,10]}
    b={'label':'a','support':2,'peak_votes':2,'region':[0,0,10,10]}
    c={'label':'b','support':20,'peak_votes':20,'region':[0,0,10,10]}
    d={'label':'a','support':20,'peak_votes':20,'region':[50,50,60,60]}
    assert select([a,b,c,d],16,.7,'region')==[a,c,d]
    assert select([b,a],1,.7,'region')==[a]
    assert select([b,a],1,None,'region')==[b,a]
    assert select([],16,.7,'region')==[]


def main(root):
    self_check()
    folder=root/'inference_ablation/results/m45_selection_v1';folder.mkdir(parents=True,exist_ok=True)
    original=root/'inference_ablation/results/m45'
    files=sorted(original.glob('shard*/records/*.json'))
    records=[json.loads(p.read_text()) for p in files]
    fixed=set(json.loads((root/'rl_val_manifest.json').read_text())['image_ids'])
    assert len(records)==len(fixed)==256 and {r['image_id'] for r in records}==fixed
    config={'protocol':PROTOCOL,'samples_per_temperature':64,'temperatures':[.7,.5],
        'minimum_support':[1,4,8,16,32],'nms_iou':.7,
        'primary':'t0.7_vote_s16_nms70','selection_has_gt_access':False,
        'claim':'COCO multi-instance adaptation repair; original author vote regions unchanged; frozen cached generations',
        'record_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    config_path=folder/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Frozen config changed')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    metadata=json.loads((root/'data/coco2017/annotations/instances_val2017.json').read_text())
    categories={c['name']:c['id'] for c in metadata['categories']}
    variants={'baseline':{r['image_id']:r['baseline'] for r in records}}
    cases=[(1,None),(1,.7)]+[(s,n) for s in [4,8,16,32] for n in [None,.7]]
    for temperature in [.7,.5]:
        for mode,key in [('vote','region'),('member','representative_box')]:
            for support,nms in cases:
                name=f't{temperature}_{mode}_s{support}_'+('nms70' if nms else 'none')
                variants[name]={r['image_id']:detections(select(r['details'][f't{temperature}']['clusters'],support,nms,key),key,categories) for r in records}
                if support==1 and nms is None:
                    for r in records:
                        geometry=lambda values:[{k:v for k,v in d.items() if k!='score'} for d in values]
                        assert geometry(variants[name][r['image_id']])==geometry(r['variants'][f't{temperature}_{mode}']),'Unfiltered geometry replay differs'
    # Freeze every output before evaluating any choice.
    for name,rows in variants.items():
        target=folder/name;target.mkdir(exist_ok=True)
        text=''.join(json.dumps({'image_id':i,'detections':rows[i]})+'\n' for i in sorted(rows))
        path=target/'predictions.jsonl'
        if path.exists() and path.read_text()!=text:raise ValueError('Replay predictions changed')
        if not path.exists():
            tmp=path.with_suffix('.tmp');tmp.write_text(text);tmp.replace(path)
    results={}
    for name,rows in variants.items():
        metrics=evaluate(metadata,rows);metrics['source']='m45_selection_v1/'+name
        metrics['image_ids']=sorted(rows);metrics['coverage']='complete';save_metrics(folder/name,metrics)
        results[name]={k:metrics[k] for k in ['images','detections','meanIoU','IoU50','IoU95','native_limit']}
        print('M45_SELECTION_RESULT',name,json.dumps(results[name]),flush=True)
    baseline=results['baseline']['meanIoU']['F1']
    lines=['# M45 cached instance-selection ablation','',PROTOCOL,'',
        'Fixed256. Primary fixed before replay: '+config['primary']+'. No validation-based winner selection. Values in percent.','',
        '| Variant | Boxes | F1 mIoU | Delta | F1@.50 | F1@.95 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,v in results.items():
        value=v['meanIoU']['F1'];lines.append(f'| {name} | {v["detections"]} | {100*value:.2f} | {100*(value-baseline):+.2f} | {100*v["IoU50"]["F1"]:.2f} | {100*v["IoU95"]["F1"]:.2f} |')
    (folder/'summary.json').write_text(json.dumps({'config':config,'results':results},indent=2)+'\n')
    (folder/'summary.md').write_text('\n'.join(lines)+'\n')
    print('M45_SELECTION_COMPLETE',config['primary'],results[config['primary']]['meanIoU'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    main(Path(parser.parse_args().root).resolve())
