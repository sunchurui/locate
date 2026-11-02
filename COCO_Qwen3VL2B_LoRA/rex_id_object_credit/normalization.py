"""Response-local object z-score (MCR-GRPO equations 8--9).

The raw credit estimator remains ours. Crowd-ignored boxes are excluded;
termination penalties are separate and never centered by this function.
"""
import math

import numpy as np


def normalize_object_credit(annotation, settings):
    mode = settings.get('object_credit_normalization', 'none')
    if mode == 'none':
        return None
    if mode != 'response_zscore':
        raise ValueError(f'Unknown object credit normalization: {mode}')
    eps = float(settings['normalization_epsilon'])
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('normalization_epsilon must be finite and positive')
    raw = [float(p.credit) for p in annotation.predictions]
    active = [p for p in annotation.predictions if not p.ignored]
    values = np.asarray([p.credit for p in active], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite raw object credit')
    mean = float(values.mean()) if len(active) else 0.
    # Population std: denominator K, not K-1, as in equation 8.
    std = float(np.sqrt(np.mean((values - mean)**2))) if len(active) else 0.
    for pred in annotation.predictions:
        if pred.ignored or len(active) <= 1 or std == 0:
            pred.credit = 0.
        else:
            pred.credit = float((pred.credit - mean) / (std + eps))
    normalized = [p.credit for p in active]
    if not all(math.isfinite(x) for x in normalized):
        raise ValueError('Nonfinite normalized object credit')
    if abs(sum(normalized)) > 1e-6 * max(1, len(active)):
        raise ValueError('Object normalization lost its zero-sum invariant')
    return {'mode': mode, 'raw_box_credit': raw, 'eligible_boxes': len(active),
            'mean': mean, 'population_std': std, 'epsilon': eps}
