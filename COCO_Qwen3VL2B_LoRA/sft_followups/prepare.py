"""Reuse pinned Detectron2 RFS, freezing its image stream to an equal update budget."""
import argparse
import ast
from collections import defaultdict
import hashlib
import itertools
import json
import math
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def sampler_class(folder):
    lock=json.loads((folder/'sources.lock.json').read_text())
    for name,item in lock.items():
        if digest(folder/'upstream'/name)!=item['sha256']:raise ValueError('Upstream source changed')
    tree=ast.parse((folder/'upstream/detectron2_sampler.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RepeatFactorTrainingSampler')
    scope={'torch':torch,'math':math,'itertools':itertools,'defaultdict':defaultdict,
        'Sampler':torch.utils.data.Sampler,'Optional':Optional,'logger':logging.getLogger('official_rfs'),
        'comm':SimpleNamespace(get_rank=lambda:0,get_world_size=lambda:1,shared_random_seed=lambda:42)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned Detectron2 RFS','exec'),scope)
    return scope['RepeatFactorTrainingSampler']


def main(root):
    folder=root/'sft_followups';out=folder/'prepared';out.mkdir(parents=True,exist_ok=True)
    config=json.loads((folder/'config.json').read_text())
    annotation=root/'data/coco2017/annotations/instances_train2017.json'
    data=json.loads(annotation.read_text());present={i['id']:set() for i in data['images']}
    for a in data['annotations']:
        if not a.get('iscrowd',0) and a['bbox'][2]>0 and a['bbox'][3]>0:present[a['image_id']].add(a['category_id'])
    rows={}
    source=root/'data/coco2017/annotations/train_rex_token.jsonl'
    for line in source.open():
        v=json.loads(line);rows[v['image_id']]=line.rstrip('\n')+'\n'
    ids=sorted(rows);assert len(ids)==118287 and set(ids)==set(present)
    datasets=[{'annotations':[{'category_id':c} for c in sorted(present[i])]} for i in ids]
    frequency={c['id']:sum(c['id'] in present[i] for i in ids)/len(ids) for c in data['categories']}
    sampler=sampler_class(folder);settings=config['class_rfs']
    t=settings['original_repeat_threshold']
    original=sampler.repeat_factors_from_category_frequency(datasets,t,sqrt=True)
    changed=float((original>1).float().mean())
    if changed<settings['fallback_if_original_changes_fewer_than_images_fraction']:
        t=float(torch.tensor(list(frequency.values()),dtype=torch.float64).median())
    factors=sampler.repeat_factors_from_category_frequency(datasets,t,sqrt=True)
    stream=sampler(factors,shuffle=True,seed=config['seed'])
    selected=[ids[index] for index in itertools.islice(iter(stream),len(ids))]
    control=json.loads((root/config['control_order']).read_text())
    assert len(selected)==len(control)==118287 and set(control)==set(ids)
    files={}
    for variant,order in [('common_order',control),('class_rfs',selected)]:
        for name,text in [(variant+'_ids.json',json.dumps(order)+'\n'),(variant+'.jsonl',''.join(rows[i] for i in order))]:
            p=out/name
            if p.exists() and p.read_text()!=text:raise ValueError('Frozen preparation differs')
            if not p.exists():p.write_text(text)
            files[name]=digest(p)
    report={'config':config,'source_rows_sha256':digest(source),'annotation_sha256':digest(annotation),
        'input_pool':len(ids),'budget_draws':len(selected),'rfs_unique_images':len(set(selected)),
        'category_image_frequencies':frequency,'original_threshold':settings['original_repeat_threshold'],
        'original_fraction_images_repeated':changed,'actual_threshold':t,
        'fraction_images_repeat_factor_above1':float((factors>1).float().mean()),
        'max_repeat_factor':float(factors.max()),'mean_repeat_factor':float(factors.mean()),
        'source_lock_sha256':digest(folder/'sources.lock.json'),'files':files,
        'shared_sources':{name:digest(root/name) for name in ['sft_ablation/baseline_train.py','sft_ablation/train.py','sft_ablation/data.py','sft_ablation/evaluate.py','sft_ablation/baseline_evaluate.py']},
        'scope':'Original Detectron2 RFS factor formula AND stochastic-repeat/shuffle stream; first118287 draws for equal-budget COCO adaptation. All GT retained per draw.'}
    path=out/'report.json'
    if path.exists() and json.loads(path.read_text())!=report:raise ValueError('Frozen report differs')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print('FOLLOWUP_DATA_READY',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    main(Path(p.parse_args().root).resolve())
