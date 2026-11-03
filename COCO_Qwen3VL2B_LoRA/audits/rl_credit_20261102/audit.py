"""Read-only audit of completed RL runs; run on a Slurm CPU allocation."""
import argparse
import ast
import contextlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    credit = root / 'rex_id_grpo_outputs_object_credit_v1'
    global_run = root / 'rex_id_grpo_outputs_global16_v3'
    cfg = json.loads((credit / 'object_credit_config.json').read_text())
    totals = Counter()
    first = []
    for line in (credit / 'object_credit_audit.jsonl').open():
        row = json.loads(line)
        a, base = row['annotation'], row['base_advantage']
        if len(first) < 32:
            first.append(row)
        n = len(a['response_ids'])
        delta = [0.] * n
        for box in a['predictions']:
            for index in box['token_indices']:
                delta[index] += cfg['box_weight'] * box['credit']
            if box['gt_index'] is None and not box['ignored']:
                totals['extra_boxes'] += 1
                totals['extra_still_positive'] += base + cfg['box_weight'] * box['credit'] > 0
        for index, value, reason in a['stops']:
            delta[index] += cfg['stop_weight'] * value
            totals['stop_penalty_sites'] += 1
        totals['valid_responses'] += 1
        totals['generated_tokens'] += n
        totals['modified_tokens'] += sum(x != 0 for x in delta)
        totals['sign_flipped_tokens'] += sum(base * (base + x) < 0 for x in delta)
        totals['base_abs_sum'] += abs(base) * n
        totals['delta_abs_sum'] += sum(abs(x) for x in delta)
        totals['base_square_sum'] += base * base * n
        totals['delta_square_sum'] += sum(x * x for x in delta)
    report = {'scope': 'Saved valid training responses only; generated-token denominators, not prompt/padding.',
              'credit_settings': cfg, 'totals': dict(totals),
              'modified_token_fraction': totals['modified_tokens'] / totals['generated_tokens'],
              'sign_flip_fraction_of_modified': totals['sign_flipped_tokens'] / totals['modified_tokens'],
              'delta_to_base_l1_ratio': totals['delta_abs_sum'] / totals['base_abs_sum'],
              'delta_to_base_l2_ratio': math.sqrt(totals['delta_square_sum'] / totals['base_square_sum']),
              'extra_box_still_positive_fraction': totals['extra_still_positive'] / totals['extra_boxes']}

    # Execute the installed Swift loss function through its real loss reduction.
    # Use actual saved advantages/credit; log-probability leaves are set equal
    # to old/reference logps (IS ratio=1). This verifies loss-level derivatives,
    # not a replay of historical model gradients or optimizer updates.
    import torch
    torch.set_num_threads(2)
    source_path = root / 'venv/lib/python3.12/site-packages/swift/rlhf_trainers/grpo_trainer.py'
    source = source_path.read_text()
    source = source.replace('safe_ratio = torch.clamp(ref_per_token_logps - per_token_logps, min=-20, max=20)',
                            'safe_ratio = ref_per_token_logps - per_token_logps')
    source = source.replace('per_token_kl = torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-10, max=10)',
                            'per_token_kl = torch.exp(safe_ratio) - safe_ratio - 1')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GRPOTrainer')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_loss_and_metrics')
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.If) and
               "self.loss_type in ['grpo', 'sapo']" in ast.unparse(n.test))
    fn.body = fn.body[:end + 1] + [ast.Return(value=ast.Name(id='loss', ctx=ast.Load()))]
    fn.returns = None
    for a in fn.args.args:
        a.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {'torch': torch, 'gather_object': lambda x: x}
    exec(compile(module, str(source_path), 'exec'), namespace)
    loss_fn = namespace['_compute_loss_and_metrics']
    length = max(len(x['annotation']['response_ids']) for x in first)
    mask = torch.zeros((len(first), length), dtype=torch.bool)
    base_adv = torch.zeros_like(mask, dtype=torch.float32)
    delta = torch.zeros_like(base_adv)
    for i, row in enumerate(first):
        a = row['annotation']; n = len(a['response_ids'])
        mask[i, :n] = True; base_adv[i, :n] = row['base_advantage']
        for box in a['predictions']:
            delta[i, box['token_indices']] += cfg['box_weight'] * box['credit']
        for index, value, _ in a['stops']:
            delta[i, index] += cfg['stop_weight'] * value
    self = NS(model=NS(training=True), compute_entropy=False, overlong_filter=False,
              beta=.02, kl_in_reward=False, rollout_importance_sampling_mode=None,
              log_rollout_offpolicy_metrics=False, disable_rollout_importance_sampling=True,
              importance_sampling_level='token', _has_teacher=False, loss_type='grpo',
              epsilon_low=.2, epsilon_high=.2, args=NS(delta=None), off_policy_sequence_mask_delta=None)
    gradients = []
    losses = []
    for adv in [base_adv, base_adv + delta]:
        logps = torch.zeros_like(adv, requires_grad=True)
        self._get_per_token_logps_and_entropies = lambda *a, **k: (logps, None)
        batch = NS(completion_mask=mask, truncated_mask=torch.zeros(len(first), dtype=torch.bool),
                   ref_per_token_logps=torch.zeros_like(adv), old_per_token_logps=torch.zeros_like(adv),
                   rollout_per_token_logps=None, advantages=adv)
        loss = loss_fn(self, None, {}, batch)
        gradients.append(torch.autograd.grad(loss, logps)[0]); losses.append(loss.item())
    diff = gradients[1] - gradients[0]
    expected = -delta / mask.sum(-1, keepdim=True) / len(first)
    assert torch.allclose(diff, expected, atol=1e-8, rtol=1e-5)
    assert diff.abs().sum() > 0 and not diff[~mask].any()
    report['loss_gradient_probe'] = {'responses': len(first), 'baseline_loss': losses[0],
        'credit_loss': losses[1], 'gradient_changed_tokens': int((diff != 0).sum()),
        'gradient_delta_l2': float(diff.norm()), 'gradient_delta_expected_max_error': float((diff-expected).abs().max()),
        'limitation': 'Installed loss core, real saved responses/advantages; controlled IS=1 logps, not historical model gradients.'}

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    predictions = []
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(root / 'data/coco2017/annotations/instances_val2017.json'))
    for folder in [global_run, credit]:
        target = folder / 'coco_val2017'
        rows = [json.loads(x) for x in (target / 'predictions.jsonl').read_text().splitlines()]
        predictions.append({x['image_id']: x['detections'] for x in rows})
        detections = [dict(d, image_id=x['image_id']) for x in rows for d in x['detections']]
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(detections), 'bbox')
            ev.params.imgIds = [x['image_id'] for x in rows]
            ev.evaluate(); ev.accumulate(); ev.summarize()
        saved = json.loads((target / 'coco_metrics.json').read_text())
        assert len(rows) == 5000 and abs(float(ev.stats[0])-saved['AP@[.50:.95]']) < 1e-12
        report[folder.name] = {'recomputed_AP': float(ev.stats[0]), 'AP50': float(ev.stats[1]),
                              'AP75': float(ev.stats[2]), 'matches_saved_metrics': True}
    a, b = predictions
    assert a.keys() == b.keys()
    report['prediction_comparison'] = {'images': len(a), 'identical_images': sum(a[k] == b[k] for k in a),
        'different_images': sum(a[k] != b[k] for k in a),
        'different_detection_counts': sum(len(a[k]) != len(b[k]) for k in a)}
    path = Path(__file__).with_name('audit_report.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
