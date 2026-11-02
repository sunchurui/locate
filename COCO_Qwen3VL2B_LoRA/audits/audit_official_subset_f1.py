"""Read-only audit of original Rex P/R/F1 sample-size dependence; no new metric."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from official_metrics import evaluate,load_predictions


def main():
    root=Path(sys.argv[1]);folder=root/'audits/official_subset_f1_20261102';folder.mkdir(parents=True,exist_ok=True)
    data=json.loads((root/'data/coco2017/annotations/instances_val2017.json').read_text())
    predictions=load_predictions(root/'evaluation/rex_id_val2017_a800_80/predictions.jsonl')
    all_ids=sorted(predictions);manifest=json.loads((root/'rl_val_manifest.json').read_text());fixed=manifest['image_ids']
    source=json.loads((root/'evaluation/rex_id_val2017_a800_80/rex_official_metrics.json').read_text())
    report={'protocol':'original Rex evaluator unchanged','full_saved_F1':source['meanIoU']['F1'],
            'fixed_subset_ids_reconstructed':False,'nested_subsets':[],'random256_subsets':[]}
    def rank(seed,i):return (hashlib.sha256(f'coco2017:val2017:seed={seed}:image_id={i}'.encode()).digest(),i)
    ordered=sorted(all_ids,key=lambda i:rank(42,i));assert sorted(ordered[:256])==fixed
    report['fixed_subset_ids_reconstructed']=True
    def save():(folder/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(ids):
        value=evaluate(data,{i:predictions[i] for i in ids})
        return {key:value[key] for key in ['images','meanIoU','IoU50','IoU95']}
    for size in [128,256,512,1000,2000,5000]:
        result=run(ordered[:size]);report['nested_subsets'].append(result);save();print('SIZE',result,flush=True)
    for seed in range(1000,1050):
        ids=sorted(all_ids,key=lambda i:rank(seed,i))[:256];result=run(ids)
        report['random256_subsets'].append({'seed':seed,'F1':result['meanIoU']['F1']})
        values=np.array([r['F1'] for r in report['random256_subsets']])
        report['random256_summary']={'completed':len(values),'mean_F1':float(values.mean()),'median_F1':float(np.median(values)),
                                     'min_F1':float(values.min()),'max_F1':float(values.max()),
                                     'fraction_above_full':float(np.mean(values>report['full_saved_F1']))}
        save()
        if len(values)%10==0:print('RANDOM',report['random256_summary'],flush=True)
    print('OFFICIAL_F1_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
