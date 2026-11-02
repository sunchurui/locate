"""Replay saved detections through unmodified Rex-Omni FastEval and metric reducers.

No generated-order, token-probability or voting-confidence scores are used.
P/R/F1 are author metrics. AP and the withdrawn P*R 'AP_set' are not reported.
"""
import importlib.util
import hashlib
import json
import math
import os
import sys
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REX_COMMIT = '6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e'
VENDOR = ROOT/'official_eval/upstream/rex/evaluation'
_AUTHOR = None


def author_module():
    global _AUTHOR
    if _AUTHOR is None:
        lock=json.loads((ROOT/'official_eval/sources.lock.json').read_text())
        for relative,digest in lock['files'].items():
            path=ROOT/'official_eval/upstream'/relative
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Original evaluator source changed: {relative}')
        sys.path.insert(0,str(VENDOR/'fastevaluate'))
        spec=importlib.util.spec_from_file_location('rex_author_metric',VENDOR/'metrics/coco_lvis_metric.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        _AUTHOR=module
    return _AUTHOR


def evaluate(dataset, predictions, image_ids=None, **unused):
    if hasattr(dataset,'dataset'):dataset=dataset.dataset
    ids=sorted(predictions if image_ids is None else image_ids);wanted=set(ids)
    if len(ids)!=len(wanted) or not wanted.issubset(predictions):raise ValueError('Duplicate/missing image rows')
    source={i['id']:i for i in dataset['images']}
    if not wanted.issubset(source):raise ValueError('Non-val image IDs')
    gt={'info':{},'images':[source[i] for i in ids],'categories':dataset['categories'],
        'annotations':[a for a in dataset['annotations'] if a['image_id'] in wanted]}
    positive=defaultdict(set)
    for a in gt['annotations']:positive[a['image_id']].add(a['category_id'])
    dt=[]
    for i in ids:
        for p in predictions[i]:
            dt.append({'image_id':i,'category_id':p['category_id'],'bbox':list(p['bbox']),'score':0.2})
    temp_root=ROOT/'runtime_tmp';temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='rex_eval_',dir=temp_root) as temporary:
        folder=Path(temporary);gt_path=folder/'gt.json';dt_path=folder/'pred.json'
        gt_path.write_text(json.dumps(gt));dt_path.write_text(json.dumps(dt))
        # The official shell script also uses GT-positive-category filtering.
        # Keep that reference separate: our frozen task queries all80 categories.
        positive_dt=[p for p in dt if p['category_id'] in positive[p['image_id']]]
        positive_path=folder/'positive.json';positive_path.write_text(json.dumps(positive_dt))
        result_path=folder/'result.json'
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--native-worker',str(gt_path),str(dt_path),str(positive_path),str(result_path)],check=True,stdout=subprocess.DEVNULL)
        native=json.loads(result_path.read_text());full=native['full'];reference=native['positive']
    return {'protocol':'rex_omni_official_fasteval_all_output_replay','images':len(ids),'detections':len(dt),
            'rex_commit':REX_COMMIT,'official_constant_score':0.2,'uses_model_confidence':False,
            'meanIoU':full['meanIoU'],'IoU50':full['IoU50'],'IoU95':full['IoU95'],
            'official_positive_only_reference':reference,'positive_only_detections':len(positive_dt),
            'task_note':'Primary retains all parsed outputs from our all80-category prompt. Separate reference reproduces official GT-positive category filtering; no prompt regeneration.',
            'implementation_note':'Unmodified author FastEval and safe_mean/f1 reducers, including internal fixed-score tie handling and maxDets. Not a permutation-invariant custom matcher.'}


def load_predictions(path):
    rows={}
    for line in Path(path).open():
        row=json.loads(line)
        if row['image_id'] in rows:raise ValueError(f'Duplicate image ID: {path}')
        rows[row['image_id']]=row['detections']
    return rows


def save_metrics(output,metrics):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    text=json.dumps(metrics,indent=2)+'\n'
    legacy=folder/'set_metrics.json'
    if legacy.exists() and not (folder/'legacy_custom_set_metrics.json').exists():
        old=json.loads(legacy.read_text())
        if old.get('protocol')=='unranked_detection_set_v1':
            (folder/'legacy_custom_set_metrics.json').write_bytes(legacy.read_bytes())
    for name in ['rex_official_metrics.json','set_metrics.json']:
        path=folder/name;tmp=folder/f'{name}.{os.getpid()}.tmp';tmp.write_text(text);tmp.replace(path)
    (folder/'evaluation_protocol.json').write_text(json.dumps({'current':'rex_official_metrics.json',
        'protocol':metrics['protocol'],'ranked_AP':'superseded','custom_AP_set':'withdrawn; not an author metric'},indent=2)+'\n')
    return folder/'rex_official_metrics.json'


def self_check():
    # Use a normal author fixture. Verify input score values do not get used.
    data={'images':[{'id':1,'width':100,'height':100}],
          'categories':[{'id':1,'name':'object'}],
          'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[0,0,10,10],'area':100,'iscrowd':0}]}
    a=evaluate(data,{1:[{'category_id':1,'bbox':[0,0,10,10],'score':.001}]})
    b=evaluate(data,{1:[{'category_id':1,'bbox':[0,0,10,10],'score':.999}]})
    for label in ['meanIoU','IoU50','IoU95']:
        assert a[label]==b[label]
        assert abs(a[label]['F1']-1)<1e-10


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--native-worker':
        author=author_module()
        def reduce_native(raw):
            out={}
            for suffix,label in [('', 'meanIoU'),('50','IoU50'),('95','IoU95')]:
                precision=author.safe_mean(raw.get('precision'+suffix,[]));recall=author.safe_mean(raw.get('recall'+suffix,[]))
                if not all(math.isfinite(x) for x in [precision,recall]):raise ValueError('Nonfinite author result')
                out[label]={'precision':precision,'recall':recall,'F1':author.f1(precision,recall)}
            return out
        gt_path,all_path,positive_path,result_path=sys.argv[2:]
        result={'full':reduce_native(author.fe.evaluate(gt_path,all_path,0,0,'coco')),
                'positive':reduce_native(author.fe.evaluate(gt_path,positive_path,0,0,'coco'))}
        Path(result_path).write_text(json.dumps(result))
    else:self_check();print('Rex official engine score-override fixture passed')
