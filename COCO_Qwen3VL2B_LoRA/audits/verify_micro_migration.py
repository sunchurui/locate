"""Regression checks for real endpoint counts and global aggregation invariants."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from micro_metrics import evaluate,load_predictions,self_check


def main():
    self_check()
    data=json.loads((ROOT/'data/coco2017/annotations/instances_val2017.json').read_text())
    rows=load_predictions(ROOT/'evaluation/rex_id_val2017_a800_80/predictions.jsonl')
    fixed=json.loads((ROOT/'rl_val_manifest.json').read_text())['image_ids']
    report={}
    for name,ids in [('full5000',sorted(rows)),('fixed256',fixed)]:
        actual=evaluate(data,rows,ids)
        audited=json.loads((ROOT/f'audits/rex_f1_root_cause_20261102/raw/{name}_initialized.json').read_text())
        for new,old in [('count_tp','audit_tp'),('count_fp','audit_fp'),('count_gt','audit_gt')]:
            assert actual['matching_counts'][new]==audited[old],(name,new)
        expected=json.loads((ROOT/'audits/rex_f1_root_cause_20261102/audit.json').read_text())['variants'][name]['E_native_match_endpoint_micro']
        for metric in ['meanIoU','IoU50','IoU95']:
            for key in ['precision','recall','F1']:
                assert abs(actual[metric][key]-expected[metric][key])<1e-12,(name,metric,key)
        report[name]={key:actual[key] for key in ['meanIoU','IoU50','IoU95','native_limit']}
    a=evaluate(data,rows,fixed[:128]);b=evaluate(data,rows,fixed[128:]);whole=evaluate(data,rows,fixed)
    for i in range(10):
        for key in ['TP','FP','GT','FN']:
            assert a['per_IoU'][i][key]+b['per_IoU'][i][key]==whole['per_IoU'][i][key],(i,key)
    # Image order and input scores must not enter endpoint aggregation.
    reversed_result=evaluate(data,{i:[dict(v,score=.999) for v in rows[i]] for i in reversed(fixed)},list(reversed(fixed)))
    assert reversed_result['per_IoU']==whole['per_IoU']
    report['checks']=['wrong/missing category zero counted','absent-GT FP counted','duplicate FP counted',
        'empty output counts FN','all 800 count entries equal audited native counts',
        'per-IoU real counts additive over disjoint image sets','image order and confidence ignored']
    (ROOT/'audits/micro_migration_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('MICRO_MIGRATION_VERIFIED',json.dumps(report),flush=True)


if __name__=='__main__':main()
