"""Causal audit of original Rex evaluator; diagnostic variants are NOT main metrics."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'audits/rex_f1_root_cause_20261102'
AUTHOR=ROOT/'official_eval/upstream/rex/evaluation'


def harmonic(p,r):return 2*p*r/(p+r) if p+r else 0.


def aggregate(raw,cat_mask,positive_only=True):
    result={}
    for suffix,label in [('', 'meanIoU'),('50','IoU50'),('95','IoU95')]:
        p=np.array(raw['precision'+suffix],dtype=float);r=np.array(raw['recall'+suffix],dtype=float)
        if positive_only:
            p=p[p>0];r=r[r>0]
        else:
            p=p[cat_mask];r=r[cat_mask];p=np.maximum(p,0);r=np.maximum(r,0)
        mp=float(p.mean()) if len(p) else 0.;mr=float(r.mean()) if len(r) else 0.
        result[label]={'precision':mp,'recall':mr,'F1':harmonic(mp,mr)}
    return result


def native_worker(engine,gt_path,dt_path,out_path):
    sys.path.insert(0,engine)
    import fastevaluate
    raw=fastevaluate.evaluate(gt_path,dt_path,0,0,'coco')
    keys=['precision','recall','precision50','recall50','precision95','recall95']
    keys += [k for k in raw if k.startswith('audit_')]
    value={k:raw[k] for k in keys}
    if not all(math.isfinite(v) for k in keys for v in value[k]):raise ValueError('Nonfinite native output')
    Path(out_path).write_text(json.dumps(value))


def call_native(engine,gt_path,dt_path,label):
    out=FOLDER/'raw'/f'{label}.json';out.parent.mkdir(exist_ok=True)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(engine),str(gt_path),str(dt_path),str(out)],check=True,stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())


def endpoint_aggregate(raw):
    k=len(raw['audit_catids'])
    gt=np.array(raw['audit_gt']).reshape(10,k)
    ep=np.array(raw['audit_endpoint_precision']).reshape(10,k)
    er=np.array(raw['audit_endpoint_recall']).reshape(10,k)
    tp=np.array(raw['audit_tp']).reshape(10,k);fp=np.array(raw['audit_fp']).reshape(10,k)
    valid=gt[0]>0
    macro={};micro={}
    for label,indices in [('meanIoU',list(range(10))),('IoU50',[0]),('IoU95',[9])]:
        p=float(ep[np.ix_(indices,valid)].mean());r=float(er[np.ix_(indices,valid)].mean())
        macro[label]={'precision':p,'recall':r,'F1':harmonic(p,r)}
        precisions=[];recalls=[]
        for t in indices:
            true=tp[t].sum();false=fp[t].sum();truth=gt[t].sum()
            precisions.append(float(true/(true+false)) if true+false else 0.)
            recalls.append(float(true/truth) if truth else 0.)
        p=float(np.mean(precisions));r=float(np.mean(recalls))
        micro[label]={'precision':p,'recall':r,'F1':harmonic(p,r)}
    return macro,micro


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/'data/coco2017/annotations/instances_val2017.json').read_text())
    predictions={}
    for line in (ROOT/'evaluation/rex_id_val2017_a800_80/predictions.jsonl').open():
        row=json.loads(line);predictions[row['image_id']]=row['detections']
    fixed=json.loads((ROOT/'rl_val_manifest.json').read_text())['image_ids']
    all_ids=sorted(predictions);images={i['id']:i for i in source['images']};cats={i['id']:i['name'] for i in source['categories']}
    engines={'original':AUTHOR/'fastevaluate','initialized':FOLDER/'native_initialized'}
    report={'official_commit':'6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e',
            'primary_evaluator_modified':False,'variants':{},'cli_parity':{},'order_tests':{},'fixtures':{}}
    def save():(FOLDER/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='inputs_',dir=FOLDER) as tmp:
        temp=Path(tmp)
        def export(name,ids,order=None):
            ids=sorted(ids);wanted=set(ids)
            order=ids if order is None else order
            gt={'info':{},'images':[images[i] for i in order],'categories':source['categories'],
                'annotations':[a for a in source['annotations'] if a['image_id'] in wanted]}
            dt=[{'image_id':i,'category_id':p['category_id'],'bbox':p['bbox'],'score':.2} for i in ids for p in predictions[i]]
            g=temp/f'{name}_gt.json';d=temp/f'{name}_pred.json';g.write_text(json.dumps(gt));d.write_text(json.dumps(dt))
            cat_ids=sorted(cats);present={a['category_id'] for a in gt['annotations'] if not a.get('iscrowd',0)}
            mask=np.array([c in present for c in cat_ids])
            return g,d,gt,dt,mask
        for name,ids in [('full5000',all_ids),('fixed256',fixed)]:
            g,d,gt,dt,mask=export(name,ids)
            original=call_native(engines['original'],g,d,name+'_original')
            initialized=call_native(engines['initialized'],g,d,name+'_initialized')
            # The diagnostic copy preserves native matching/recall exactly.
            for key in ['recall','recall50','recall95']:
                assert np.array_equal(original[key],initialized[key]),key
            macro,micro=endpoint_aggregate(initialized)
            report['variants'][name]={
                'A_original':aggregate(original,mask,True),
                'B_original_raw_zero_inclusive':aggregate(original,mask,False),
                'C_initialized_zero_inclusive':aggregate(initialized,mask,False),
                'D_native_match_endpoint_macro':macro,
                'E_native_match_endpoint_micro':micro}
            print('VARIANTS',name,json.dumps({k:v['meanIoU'] for k,v in report['variants'][name].items()}),flush=True);save()

            # Execute the original conversion script and CLI, not our wrapper.
            answer=temp/f'{name}_answer.jsonl'
            with answer.open('w') as f:
                for i in sorted(ids):
                    extracted={}
                    for p in predictions[i]:
                        x,y,w,h=p['bbox'];extracted.setdefault(cats[p['category_id']],[]).append([x,y,x+w,y+h])
                    f.write(json.dumps({'image_path':images[i]['file_name'],'extracted_predictions':extracted})+'\n')
            tsv=temp/f'{name}.tsv';env=dict(os.environ,PYTHONPATH=str(engines['original']))
            converter=[sys.executable,str(AUTHOR/'utils/convert_coco_lvis_to_standard_format.py'),
                       '--our_pred_jsonl',str(answer),'--coco_json',str(g),'--out_tsv',str(tsv),'--positive_only']
            converted=subprocess.run(converter,env=env,capture_output=True,text=True,check=True)
            (FOLDER/f'{name}_converter.log').write_text(converted.stdout+converted.stderr)
            cli=[sys.executable,str(AUTHOR/'metrics/coco_lvis_metric.py'),'--gt',str(g),'--pred_tsv',str(tsv),'--eval_type','coco']
            done=subprocess.run(cli,env=env,capture_output=True,text=True,check=True)
            (FOLDER/f'{name}_official_cli.log').write_text(done.stdout+done.stderr)
            tsv_raw=call_native(engines['original'],g,tsv,name+'_official_tsv')
            mapping={c['name']:c['id'] for c in gt['categories']};converted_dt=[]
            for line in tsv.read_text().splitlines():
                i,entries=line.split('\t',1)
                for p in json.loads(entries):converted_dt.append({'image_id':int(i),'category_id':mapping[p['class']],
                                                                 'bbox':p['rect'],'score':p['conf']})
            jd=temp/f'{name}_converted.json';jd.write_text(json.dumps(converted_dt))
            json_raw=call_native(engines['original'],g,jd,name+'_official_converted_json')
            differences={key:float(np.max(np.abs(np.array(tsv_raw[key])-np.array(json_raw[key])))) for key in ['precision','recall','precision50','recall50','precision95','recall95']}
            result=aggregate(tsv_raw,mask,True)
            report['cli_parity'][name]={'original_converter_and_cli_executed':True,'json_vs_tsv_max_differences':differences,
                'official_positive_only_result':result,'detections_after_official_filter':len(converted_dt),
                'command':cli,'cli_avg_f1':float(re.search(r'Avg F1\s*\|\s*([0-9.]+)',done.stdout).group(1))}
            assert abs(report['cli_parity'][name]['cli_avg_f1']-result['meanIoU']['F1'])<.000051
            print('CLI_PARITY',name,report['cli_parity'][name],flush=True);save()

            # All boxes and per-image matches are identical; only GT image order changes.
            orders={'original_dataset_order':[i['id'] for i in source['images'] if i['id'] in set(ids)]}
            for seed in [101,102,103,104]:
                values=list(sorted(ids));np.random.default_rng(seed).shuffle(values);orders[f'shuffle{seed}']=values
            tests={}
            for label,order in orders.items():
                og,od,_,_,om=export(name+'_'+label,ids,order)
                value=call_native(engines['original'],og,od,name+'_'+label)
                for key in ['recall','recall50','recall95']:assert np.array_equal(original[key],value[key]),'Image order changed matching recall'
                tests[label]=aggregate(value,om,True)
            report['order_tests'][name]=tests;save()

        # Minimal reproductions: one class succeeds, another completely fails;
        # and a box that matches .50:.90 but fails .95.
        toy={'info':{},'images':[{'id':1,'width':100,'height':100}],
             'categories':[{'id':1,'name':'a'},{'id':2,'name':'b'}],
             'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[0,0,10,10],'area':100,'iscrowd':0},
                            {'id':2,'image_id':1,'category_id':2,'bbox':[20,0,10,10],'area':100,'iscrowd':0}]}
        for name,box in [('class_b_all_wrong',[60,60,10,10]),('class_b_fails_only_095',[20,0,11,10])]:
            g=temp/(name+'_gt.json');d=temp/(name+'_pred.json');g.write_text(json.dumps(toy))
            d.write_text(json.dumps([{'image_id':1,'category_id':1,'bbox':[0,0,10,10],'score':.2},
                                     {'image_id':1,'category_id':2,'bbox':box,'score':.2}]))
            values={}
            for label,engine in engines.items():
                raw=call_native(engine,g,d,name+'_'+label)
                values[label]={'raw':raw,'author_aggregation':aggregate(raw,np.array([True,True]),True),
                               'zero_inclusive_diagnostic':aggregate(raw,np.array([True,True]),False)}
            report['fixtures'][name]=values;save();print('FIXTURE',name,json.dumps(values),flush=True)
    print('ROOT_CAUSE_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':native_worker(*sys.argv[2:])
    else:main()
