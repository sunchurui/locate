"""Fresh filtered-GT COCO checks, sample-size curve, and exact AP counterexample."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(dataset, predictions, wanted):
    wanted = set(wanted)
    # Construct a fresh GT object containing only selected images/annotations.
    # This independently checks the earlier all-GT + params.imgIds route.
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {'images': [dict(i) for i in dataset['images'] if i['id'] in wanted],
                      'annotations': [dict(a) for a in dataset['annotations'] if a['image_id'] in wanted],
                      'categories': dataset['categories'], 'info': {}}
        gt.createIndex()
        dt = gt.loadRes([dict(d, image_id=i) for i in sorted(wanted) for d in predictions[i]])
        evaluator = COCOeval(gt, dt, 'bbox'); evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return {'AP': float(evaluator.stats[0]*100), 'AP50': float(evaluator.stats[1]*100),
            'AR100': float(evaluator.stats[8]*100), 'images': len(wanted)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); args = p.parse_args()
    root = Path(args.root); folder = root/'inference_ablation/results/subset_audit_20261102'
    dataset = json.loads((root/'data/coco2017/annotations/instances_val2017.json').read_text())
    predictions = {}
    for line in (root/'evaluation/rex_id_val2017_a800_80/predictions.jsonl').open():
        row = json.loads(line)
        if row['image_id'] in predictions:raise ValueError('Duplicate prediction image')
        predictions[row['image_id']] = row['detections']
    all_ids = sorted(i['id'] for i in dataset['images'])
    assert set(all_ids) == set(predictions)
    key = lambda i: (hashlib.sha256(f'coco2017:val2017:seed=42:image_id={i}'.encode()).digest(),i)
    ordered = sorted(all_ids,key=key)
    report = {'note': 'Read-only diagnostics: no new model inference; fresh filtered GT per evaluation.',
              'size_curve': []}
    def save():
        (folder/'ap_mechanism.json').write_text(json.dumps(report,indent=2)+'\n')

    # Same category and a real GT in both images. All IoUs for true boxes are1.
    toy = {'images': [{'id':i,'width':100,'height':100} for i in [1,2]],
           'categories': [{'id':1,'name':'object'}],
           'annotations': [{'id':i,'image_id':i,'category_id':1,'bbox':[0,0,10,10],
                            'area':100,'iscrowd':0} for i in [1,2]]}
    toy_predictions = {i: [{'category_id':1,'bbox':[0,0,10,10],'score':a},
                           {'category_id':1,'bbox':[20,20,10,10],'score':b}]
                       for i,a,b in [(1,.9,.8),(2,.7,.6)]}
    report['same_class_counterexample'] = {name:evaluate(toy,toy_predictions,ids)
            for name,ids in [('image1',[1]),('image2',[2]),('both',[1,2])]}
    save(); print('COUNTEREXAMPLE',report['same_class_counterexample'],flush=True)

    for size in [128,256,512,1000,2000,4000,5000]:
        result = evaluate(dataset,predictions,ordered[:size])
        report['size_curve'].append(result); save(); print('SIZE',result,flush=True)

    report['tie_breaking_diagnostics'] = []
    for seed in [20261102,20261103]:
        # Perturb only score ties. Adjacent distinct baseline scores differ by.001;
        # each perturbation is less than1e-6. Never used as a replacement baseline.
        changed = {}
        for i, boxes in predictions.items():
            changed[i] = []
            for rank, box in enumerate(boxes):
                integer = int.from_bytes(hashlib.sha256(f'{seed}:{i}:{rank}'.encode()).digest()[:8],'big')
                changed[i].append(dict(box,score=box['score']+1e-6*integer/2**64))
        result = {'seed':seed,'full5000':evaluate(dataset,changed,all_ids),
                  'fixed256':evaluate(dataset,changed,ordered[:256])}
        report['tie_breaking_diagnostics'].append(result);save();print('TIE_DIAGNOSTIC',result,flush=True)
    print('MECHANISM_AUDIT_COMPLETE',flush=True)


if __name__ == '__main__':main()
