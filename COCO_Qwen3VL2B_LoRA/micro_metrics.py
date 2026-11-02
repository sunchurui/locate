"""Global endpoint micro P/R/F1 with Rex matching; user authorized 2026-11-02.

Pool TP/FP/FN across images/categories per IoU (.50:.05:.95). meanIoU F1 is
harmonic(mean(P_t),mean(R_t)), exactly audited variant E, not original Rex F1.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT/'official_eval/counts_engine'
PROTOCOL = 'rex_matching_global_endpoint_micro_v1'
REX_COMMIT = '6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e'
IOU_THRESHOLDS = [round(.5+.05*i,2) for i in range(10)]


def harmonic(p,r):
    return 2*p*r/(p+r) if p+r else 0.


def engine_manifest():
    path = ENGINE/'counts_manifest.json'
    if not path.exists():
        raise RuntimeError('Build first: python official_eval/build_counts_engine.py')
    manifest = json.loads(path.read_text())
    for relative,sha in manifest['files'].items():
        if hashlib.sha256((ENGINE/relative).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'Endpoint engine changed: {relative}')
    if hashlib.sha256((ROOT/'official_eval/build_counts_engine.py').read_bytes()).hexdigest()!=manifest['builder_sha256']:
        raise ValueError('Counts builder and installed build differ')
    return manifest


def summarize_counts(raw):
    k = len(raw['count_catids'])
    if not k or any(len(raw[key])!=10*k for key in ('count_tp','count_fp','count_gt')):
        raise ValueError('Expected 10 IoUs by category native counts')
    per_iou=[]
    for t,threshold in enumerate(IOU_THRESHOLDS):
        values={}
        for key,native in [('TP','count_tp'),('FP','count_fp'),('GT','count_gt')]:
            vector=raw[native][t*k:(t+1)*k]
            if any(not math.isfinite(v) or v<0 or v!=round(v) for v in vector):
                raise ValueError('Non-integer/negative native counts')
            values[key]=int(sum(vector))
        tp,fp,gt=values['TP'],values['FP'],values['GT']
        if tp>gt:raise ValueError('TP exceeds eligible GT')
        p=tp/(tp+fp) if tp+fp else 0.
        r=tp/gt if gt else 0.
        per_iou.append({'IoU':threshold,**values,'FN':gt-tp,'precision':p,'recall':r,'F1':harmonic(p,r)})
    p=sum(v['precision'] for v in per_iou)/10
    r=sum(v['recall'] for v in per_iou)/10
    point=lambda v:{key:v[key] for key in ('precision','recall','F1')}
    return {'meanIoU':{'precision':p,'recall':r,'F1':harmonic(p,r)},
        'IoU50':point(per_iou[0]),'IoU95':point(per_iou[-1]),'per_IoU':per_iou}


def evaluate(dataset,predictions,image_ids=None,**unused):
    if hasattr(dataset,'dataset'):dataset=dataset.dataset
    ids=sorted(predictions if image_ids is None else image_ids);wanted=set(ids)
    if not ids or len(ids)!=len(wanted) or not wanted.issubset(predictions):
        raise ValueError('Empty/duplicate/missing image rows')
    images={v['id']:v for v in dataset['images']}
    if not wanted.issubset(images):raise ValueError('Unknown image IDs')
    categories={v['id'] for v in dataset['categories']}
    gt={'info':{},'images':[images[i] for i in ids],'categories':dataset['categories'],
        'annotations':[v for v in dataset['annotations'] if v['image_id'] in wanted]}
    detections=[];counts=Counter()
    for i in ids:
        for v in predictions[i]:
            box=list(v['bbox'])
            if v['category_id'] not in categories or len(box)!=4 or not all(math.isfinite(x) for x in box):
                raise ValueError('Invalid category or box')
            detections.append({'image_id':i,'category_id':v['category_id'],'bbox':box,'score':.2})
            counts[i,v['category_id']]+=1
    manifest=engine_manifest()
    temp_root=ROOT/'runtime_tmp';temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='micro_eval_',dir=temp_root) as tmp:
        folder=Path(tmp);g,d,out=folder/'gt.json',folder/'dt.json',folder/'counts.json'
        g.write_text(json.dumps(gt));d.write_text(json.dumps(detections))
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--native-worker',str(g),str(d),str(out)],
                       check=True,stdout=subprocess.DEVNULL)
        raw=json.loads(out.read_text())
    return {'protocol':PROTOCOL,'images':len(ids),'detections':len(detections),
        'rex_commit':REX_COMMIT,'official_constant_score':.2,'uses_model_confidence':False,
        **summarize_counts(raw),'matching_counts':raw,
        'native_limit':{'max_detections_per_image_category':100,'observed_max':max(counts.values(),default=0),
            'detections_above_cap':sum(max(0,v-100) for v in counts.values())},
        'counts_engine_manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        'definition':'Per IoU pool TP/FP/FN across all images/categories. P=TP/(TP+FP), R=TP/(TP+FN). meanIoU F1=harmonic(mean(P_t),mean(R_t)), not mean(F1_t). Zero denominators return0.',
        'implementation_note':'User-authorized project metric, not original Rex reported F1. Native matching/crowd/ignore/maxDets/fixed-score ties retained; no PR reduction or positive-category filtering.'}


def load_predictions(path):
    rows={}
    with Path(path).open() as stream:
        for line in stream:
            value=json.loads(line)
            if value['image_id'] in rows:raise ValueError(f'Duplicate image ID: {path}')
            rows[value['image_id']]=value['detections']
    return rows


def save_metrics(output,metrics):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    old=folder/'set_metrics.json'
    if old.exists() and json.loads(old.read_text()).get('protocol')!=PROTOCOL:
        archive=folder/'legacy_metrics';archive.mkdir(exist_ok=True)
        sha=hashlib.sha256(old.read_bytes()).hexdigest()[:12];target=archive/f'set_metrics_{sha}.json'
        if not target.exists():target.write_bytes(old.read_bytes())
    for name in ['micro_metrics.json','set_metrics.json']:
        target=folder/name;tmp=folder/f'{name}.{os.getpid()}.tmp'
        tmp.write_text(json.dumps(metrics,indent=2)+'\n');tmp.replace(target)
    (folder/'evaluation_protocol.json').write_text(json.dumps({'current':'micro_metrics.json','protocol':PROTOCOL,
        'original_rex_f1':'historical reference; upstream aggregation issues; not current',
        'ranked_AP':'superseded','custom_AP_set':'withdrawn'},indent=2)+'\n')
    return folder/'micro_metrics.json'


def self_check():
    data={'images':[{'id':1,'width':100,'height':100}],
        'categories':[{'id':1,'name':'a'},{'id':2,'name':'b'},{'id':3,'name':'absent'}],
        'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[0,0,10,10],'area':100,'iscrowd':0},
                       {'id':2,'image_id':1,'category_id':2,'bbox':[20,0,10,10],'area':100,'iscrowd':0}]}
    good={'category_id':1,'bbox':[0,0,10,10]};bad={'category_id':2,'bbox':[60,60,10,10]}
    result=evaluate(data,{1:[good,bad]})
    assert result['IoU50']=={'precision':.5,'recall':.5,'F1':.5}
    result=evaluate(data,{1:[good,bad,{'category_id':3,'bbox':[0,0,10,10]}]})
    assert result['per_IoU'][0]['FP']==2 and abs(result['IoU50']['F1']-.4)<1e-12
    assert evaluate(data,{1:[]})['IoU50']['F1']==0
    result=evaluate(data,{1:[dict(good,score=.99),dict(good,score=.01)]})
    assert result['per_IoU'][0]['TP']==1 and result['per_IoU'][0]['FP']==1
    print('MICRO_FIXTURES_PASSED',flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--native-worker':
        engine_manifest();sys.path.insert(0,str(ENGINE))
        import fastevaluate
        raw=fastevaluate.evaluate(sys.argv[2],sys.argv[3],0,0,'coco')
        Path(sys.argv[4]).write_text(json.dumps({k:v for k,v in raw.items() if k.startswith('count_')}))
    else:self_check()
