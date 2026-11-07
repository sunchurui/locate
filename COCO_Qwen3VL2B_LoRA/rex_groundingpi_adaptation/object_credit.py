"""Opt-in record-level GRPO credit for Rex marker outputs; not a reward function."""
import math
import re
from collections import Counter

import torch
from scipy.optimize import linear_sum_assignment

from ..configs.groundingpi_reward import _parse_truth
from ..configs.rex_format import parse_rex_boxes

REF_START = '<|object_ref_start|>'
REF_END = '<|object_ref_end|>'
BOX_START = '<|box_start|>'
BOX_END = '<|box_end|>'
NUMBER = re.compile(r'<(\d+)>')
COCO80 = frozenset('person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush'.split('|'))


def box_iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    aarea = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    barea = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = aarea + barea - inter
    return inter / union if union else 0.


def _token_ids(tokenizer):
    mapping = {s: tokenizer.convert_tokens_to_ids(s) for s in (REF_START, REF_END, BOX_START, BOX_END)}
    if len(set(mapping.values())) != 4 or any(v is None or v == tokenizer.unk_token_id for v in mapping.values()):
        raise ValueError('object credit requires four distinct, registered Rex marker tokens')
    return mapping


def parse_ids(ids, tokenizer):
    """Use original generated IDs, never retokenize decoded coordinates."""
    markers = _token_ids(tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    entries, boxes, used = [], [], set()
    for start, token_id in enumerate(ids):
        if token_id != markers[REF_START]:
            continue
        try:
            ref_end = ids.index(markers[REF_END], start + 1)
            box_start = ids.index(markers[BOX_START], ref_end + 1)
            box_end = ids.index(markers[BOX_END], box_start + 1)
        except ValueError:
            return None
        if any(k in used for k in range(start, box_end + 1)):
            return None
        if any(ids[k] in markers.values() for k in range(start + 1, ref_end)):
            return None
        if tokenizer.decode(ids[ref_end + 1:box_start], skip_special_tokens=False).strip():
            return None
        label = tokenizer.decode(ids[start + 1:ref_end], skip_special_tokens=False).strip().lower()
        if not label:
            return None
        payload_positions = list(range(box_start + 1, box_end))
        nums = [(i, int(m.group(1))) for i in payload_positions if (m := NUMBER.fullmatch(tokens[i]))]
        nonnum = [ids[i] for i in payload_positions if not NUMBER.fullmatch(tokens[i])]
        other = tokenizer.decode(nonnum, skip_special_tokens=False).strip()
        if other != 'None' and other.strip(' ,\n\r\t'):
            return None
        if other == 'None' and nums:
            return None
        if len(nums) % 4:
            return None
        entry = {'label': label, 'category_tokens': list(range(start + 1, ref_end)), 'end_token': box_end}
        for j in range(0, len(nums), 4):
            coords = [x[1] for x in nums[j:j+4]]
            x1, y1, x2, y2 = coords
            valid = 0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999
            # GT plugin represents coordinates in 0..1000, Rex coordinate bins in 0..999.
            boxes.append({'label': label, 'box': [v * 1000. / 999. for v in coords],
                          'tokens': [x[0] for x in nums[j:j+4]], 'valid': valid})
        entries.append(entry)
        used.update(range(start, box_end + 1))
    if any(ids[i] in markers.values() and i not in used for i in range(len(ids))):
        return None
    if not entries:
        # A 'None' answer and truncations have no localizable category/box markers.
        return ([], []) if tokenizer.decode(ids, skip_special_tokens=True).strip() == 'None' else None
    return entries, boxes


def credit_for_response(ids, truth, tokenizer, complete_gt=False, allowed_labels=None):
    response_text = tokenizer.decode(ids, skip_special_tokens=False)
    expected_boxes, format_valid = parse_rex_boxes(response_text)
    if not format_valid:
        return {}, Counter({'unparsed': 1})
    parsed = parse_ids(ids, tokenizer)
    metrics = Counter()
    if parsed is None:
        metrics['unparsed'] += 1
        return {}, metrics
    entries, boxes = parsed
    observed_boxes = [(box['label'], box['box']) for box in boxes if box['valid']]
    if observed_boxes != expected_boxes:
        return {}, Counter({'alignment_mismatch': 1})
    gt = _parse_truth(truth)
    credit = {}
    matches = {}
    valid_indices = [i for i, p in enumerate(boxes) if p['valid']]
    if valid_indices and gt:
        scores = [[box_iou(boxes[i]['box'], box) if boxes[i]['label'] == label else 0.
                   for label, box in gt] for i in valid_indices]
        rows, cols = linear_sum_assignment([[-v for v in row] for row in scores])
        for row, col in zip(rows, cols):
            if scores[row][col] > 0.:
                matches[valid_indices[row]] = (int(col), float(scores[row][col]))
    for i, box in enumerate(boxes):
        if not box['valid']:
            metrics['invalid_geometry'] += 1
            continue
        if i in matches:
            local = matches[i][1]
            metrics['matched'] += 1
        elif complete_gt:
            local = -1.
            metrics['unmatched'] += 1
        else:
            metrics['unmatched_unknown'] += 1
            continue
        for token in box['tokens']:
            credit[token] = local
    used_gt = {match[0] for match in matches.values()}
    if allowed_labels is not None:
        for entry in entries:
            if entry['label'] not in allowed_labels:
                for token in entry['category_tokens']:
                    credit[token] = -1.
                metrics['unrecognized_category'] += 1
    # Only a true EOS is a stopping decision. Max-length truncation is not.
    eos_ids = {x for x in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>'))
               if x is not None and x != tokenizer.unk_token_id}
    ended = bool(ids) and ids[-1] in eos_ids
    if complete_gt and ended:
        last_by_label = {entry['label']: entry for entry in entries}
        for label, entry in last_by_label.items():
            gt_indices = [j for j, (glabel, _) in enumerate(gt) if glabel == label]
            if gt_indices:
                missed = sum(j not in used_gt for j in gt_indices)
                if missed:
                    credit[entry['end_token']] = -missed / len(gt_indices)
                    metrics['category_stop_missed'] += missed
        # Missing categories have no category end marker; attribute them to actual EOS.
        missing_absent = sum(j not in used_gt and label not in last_by_label for j, (label, _) in enumerate(gt))
        if gt and missing_absent:
            credit[len(ids) - 1] = -missing_absent / len(gt)
            metrics['absent_category_missed'] += missing_absent
    elif not ended:
        metrics['not_ended'] += 1
    metrics['parsed'] += 1
    return credit, metrics


def apply_object_credit(data, tokenizer, weight, complete_gt=False, allowed_labels=None):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('object credit weight must be finite and nonnegative')
    responses, mask = data.batch['responses'], data.batch['response_mask']
    base = data.batch['advantages']
    if responses.shape != mask.shape or base.shape != mask.shape:
        raise ValueError('response, mask, advantage shapes differ')
    if 'ground_truth' not in data.non_tensor_batch:
        raise ValueError('object credit requires ground_truth for each response')
    adjustment = torch.zeros_like(base)
    metrics = Counter()
    for row in range(len(data)):
        length = int(mask[row].sum().item())
        if not torch.all(mask[row, :length]) or torch.any(mask[row, length:]):
            raise ValueError('object credit requires right-padded response masks')
        ids = responses[row, :length].tolist()
        local, row_metrics = credit_for_response(ids, data.non_tensor_batch['ground_truth'][row],
                                                tokenizer, complete_gt, allowed_labels)
        for position, value in local.items():
            adjustment[row, position] = value * weight
        metrics.update(row_metrics)
    data.batch['advantages'] = base + adjustment
    # Keep returns as produced by GRPO. They are not used as actor advantages.
    metrics['adjusted_tokens'] = int(torch.count_nonzero(adjustment).item())
    metrics['abs_adjustment_sum'] = float(adjustment.abs().sum().item())
    return {f'object_credit/{k}': float(v) for k, v in metrics.items()}
