"""CANL coordinate-token mean confidence: frozen inference only."""
from common import FIRST, LAST, PROMPT, detections, parse


def run(engine, image, info, baseline, config, upstream, categories):
    settings = config['methods']['m06']
    bank = engine.generate(image, PROMPT, f'm06:{info["id"]}',
                           samples=settings['samples'], temperature=settings['temperature'],
                           top_k=settings['top_k'], top_p=settings['top_p'])
    boxes = [parse(item, engine.tokenizer, info['width'], info['height']) for item in bank]
    confidence = []
    for item in bank:
        values = [p for token, p in zip(item['token_ids'], item['coordinate_probabilities']) if FIRST <= token <= LAST]
        # Author's reduction: sum of chosen coordinate-token probabilities / count.
        confidence.append(sum(values) / len(values) if values else 0.)
    variants = {'greedy_confidence': detections(baseline, categories, confidence=True),
                'sample0_order': detections(boxes[0], categories)}
    choices = {}
    for budget in settings['budgets']:
        best = max(range(budget), key=lambda i: (confidence[i], -i))
        choices[str(budget)] = {'candidate_index': best, 'coordinate_confidence': confidence[best]}
        variants[f'best{budget}_order'] = detections(boxes[best], categories)
        variants[f'best{budget}_confidence'] = detections(boxes[best], categories, confidence=True)
    return variants, {'choices': choices, 'all_confidences': confidence,
                      'generation_temperature': settings['temperature'], 'samples': len(bank)}
