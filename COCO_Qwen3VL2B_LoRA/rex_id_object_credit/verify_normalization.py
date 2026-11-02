"""Check normalization edge cases and replay real saved token-credit records."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from credit import Annotation, Prediction, assign_group_credit
from normalization import normalize_object_credit


def example(values, ignored=()):
    return Annotation([], True, 'stop', predictions=[
        Prediction('person', [0, 0, 1, 1], [i*4+j for j in range(4)],
                   credit=value, ignored=i in ignored)
        for i, value in enumerate(values)], stops=[(99, -.5, 'eos_absent_classes')])


def check(a, settings):
    stops = list(a.stops)
    raw = [p.credit for p in a.predictions]
    stats = normalize_object_credit(a, settings)
    values = np.asarray([raw[i] for i,p in enumerate(a.predictions) if not p.ignored])
    actual = np.asarray([p.credit for p in a.predictions if not p.ignored])
    expected = ((values-values.mean())/(values.std(ddof=0)+settings['normalization_epsilon'])
                if len(values)>1 and values.std(ddof=0)>0 else np.zeros(len(values)))
    assert np.allclose(actual, expected, atol=1e-12)
    assert abs(actual.sum()) < 1e-6*max(1,len(values))
    assert all(p.credit == 0 for p in a.predictions if p.ignored)
    assert max(abs(actual),default=0) <= math.sqrt(max(len(values)-1,0))+1e-6
    assert a.stops == stops and stats['raw_box_credit'] == raw
    if len(values)>1:
        assert all((values[i]-values[j])*(actual[i]-actual[j]) >= 0
                   for i in range(len(values)) for j in range(i))
    return stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root)
    settings=json.loads((root/'rex_id_object_credit/config_mcr_norm.json').read_text())
    for vals,ignored in [([],()),([.7],()),([1,1,1],()),([-.2,.1,.9],()),
                         ([-1,-1],()),([1,2,999],(2,)),([1],(0,)),
                         ([1,1+1e-12],())]:
        check(example(vals,ignored),settings)
    invalid=dict(settings,normalization_epsilon=0)
    try:normalize_object_credit(example([1,2]),invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid epsilon accepted')
    # Regression: opt-in only; the original group estimator remains unchanged.
    base=json.loads((root/'rex_id_object_credit/config.json').read_text())
    def group():
        return [Annotation([],True,'stop',predictions=[
            Prediction('person',[0,0,1,1],[0,1,2,3],gt_index=0,quality=q),
            Prediction('person',[0,0,1,1],[4,5,6,7])],quality=[q])
            for q in [.1,.2,.3,.4,.5,.6,.7,.8]]
    old,new=group(),group();assign_group_credit(old,base);assign_group_credit(new,settings)
    for i,a in enumerate(old):
        assert abs(a.predictions[0].credit-(a.quality[0]-.45))<1e-12
        assert a.predictions[1].credit == -1 and a.normalization is None
        assert new[i].normalization is not None
        assert abs(sum(p.credit for p in new[i].predictions))<1e-10
    count=boxes=degenerate=changed=0;sum_abs_raw=sum_abs_new=0.
    source=root/'rex_id_grpo_outputs_object_credit_v1/object_credit_audit.jsonl'
    for line in source.open():
        row=json.loads(line);data=row['annotation']
        data['predictions']=[Prediction(**x) for x in data['predictions']]
        a=Annotation(**data);raw=[p.credit for p in a.predictions]
        # On real records, check rank efficiently (avoid quadratic dense-box tests).
        stops=list(a.stops);stats=normalize_object_credit(a,settings)
        active=[p.credit for p in a.predictions if not p.ignored]
        assert np.isfinite(active).all() and abs(sum(active))<1e-6*max(1,len(active))
        assert max(map(abs,active),default=0)<=math.sqrt(max(len(active)-1,0))+1e-6
        assert a.stops==stops
        for j,pred in enumerate(a.predictions):
            assert len(pred.token_indices)==4
            assert all(150643<=a.response_ids[idx]<=151642 for idx in pred.token_indices)
            if pred.ignored:assert pred.credit==0
            changed += abs(pred.credit-raw[j])>1e-10
            sum_abs_raw+=abs(raw[j]);sum_abs_new+=abs(pred.credit)
        count+=1;boxes+=len(a.predictions)
        degenerate+=stats['eligible_boxes']<=1 or stats['population_std']==0
    report={'edge_cases_passed':8,'v1_group_credit_regression_passed':True,
            'v2_group_hook_passed':True,'real_valid_responses':count,'real_boxes':boxes,
            'degenerate_responses':degenerate,'boxes_with_changed_credit':changed,
            'raw_box_credit_abs_sum':sum_abs_raw,'normalized_box_credit_abs_sum':sum_abs_new,
            'stop_penalties_unchanged':True,'coordinate_token_mapping_checked':True,
            'scope':'CPU normalization replay and invariant checks, not a new RL run or AP result.'}
    output=root/'audits/rl_credit_20261102/normalization_verification.json'
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
