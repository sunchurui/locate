"""Existing GroundingPI-inspired reward, adapted to Rex marker answers."""
from verl.configs.groundingpi_reward import _parse_truth, _set_reward, _strict_reward
from verl.configs.rex_format import parse_rex_boxes


def compute_score(predictions, ground_truths):
    if len(predictions) != len(ground_truths):
        raise ValueError('prediction and reference batch lengths differ')
    scores = []
    for text, truth_raw in zip(predictions, ground_truths):
        pred, valid = parse_rex_boxes(text)
        truth = _parse_truth(truth_raw)
        set_score = _set_reward(pred, truth) if valid else 0.0
        strict_score = _strict_reward(pred, truth, valid)
        scores.append({'overall': 0.7 * set_score + 0.3 * strict_score,
                       'groundingpi_set': set_score, 'groundingpi_strict': strict_score,
                       'rex_format_valid': float(valid)})
    return scores
