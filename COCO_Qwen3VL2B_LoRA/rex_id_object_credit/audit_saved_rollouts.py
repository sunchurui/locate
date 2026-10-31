"""Audit credit on real saved rollouts before GPU execution.

Saved completions are decoded text, not raw generation IDs. This audit checks
real response parsing/matching and coordinate tokenization; only the runtime
integration can verify alignment with the actual sampled IDs and loss mask.
No EOS is synthesized, and no EOS penalty is claimed by this offline audit.
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from credit import ObjectAnnotator, assign_group_credit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--rollouts', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    sys.path.insert(0, str(root / 'rex_id_grpo'))
    from reward_plugin import core, parse
    settings = json.loads((root / 'rex_id_object_credit/config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(root / 'rex_id_rl_base', local_files_only=True)
    annotator = ObjectAnnotator(tokenizer, parse, settings)
    metadata = json.loads((root / 'rex_id_object_credit/ignore_metadata.json').read_text())['images']
    rows = [json.loads(line) for line in (root / 'data/coco2017/annotations/train_rex_id_grpo_epoch.jsonl').open()]
    report = {'source': str(args.rollouts), 'scope': 'real saved text retokenized; actual generation alignment checked at runtime',
              'responses': 0, 'valid_responses': 0, 'boxes': 0, 'matched_boxes': 0, 'extra_boxes': 0,
              'ignored_boxes': 0, 'class_stop_penalties': 0, 'positive_box_credit': 0, 'negative_box_credit': 0,
              'steps': []}
    examples = []
    for line in Path(args.rollouts).open():
        saved = json.loads(line)
        step = int(saved['step'][0])
        if len(saved['completion']) != 256 or set(saved['step']) != {str(step)}:
            raise ValueError('Saved rollout is not a complete update')
        ordered = rows[(step - 1) * 32:step * 32]
        report['steps'].append(step)
        for group, row in enumerate(ordered):
            truth = core._parse_truth(row['ground_truth'])
            annotations = []
            for text in saved['completion'][group * 8:(group + 1) * 8]:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if annotator.decode(ids) != text:
                    raise ValueError('Saved decoded text does not round-trip through the actual tokenizer')
                annotation = annotator.annotate(ids, 'stop', truth, metadata[str(row['image_id'])])
                annotations.append(annotation)
                # Deliberately does not append EOS: finish reason was not saved.
            assign_group_credit(annotations, settings)
            for annotation in annotations:
                report['responses'] += 1
                report['valid_responses'] += annotation.valid
                report['boxes'] += len(annotation.predictions)
                report['class_stop_penalties'] += len(annotation.stops)
                if any(reason == 'eos_absent_classes' for _, _, reason in annotation.stops):
                    raise ValueError('Offline audit must not claim an unsaved EOS')
                used = [p.gt_index for p in annotation.predictions if p.gt_index is not None]
                if len(used) != len(set(used)):
                    raise ValueError('GT was matched more than once')
                for pred in annotation.predictions:
                    if len(pred.token_indices) != 4:
                        raise ValueError('Box credit is not mapped to four coordinate tokens')
                    if not -1 <= pred.credit <= 1:
                        raise ValueError('Object credit outside bounded range')
                    report['matched_boxes'] += pred.gt_index is not None
                    report['extra_boxes'] += pred.gt_index is None and not pred.ignored
                    report['ignored_boxes'] += pred.ignored
                    report['positive_box_credit'] += pred.credit > 0
                    report['negative_box_credit'] += pred.credit < 0
                if len(examples) < 12 and annotation.stops:
                    examples.append({'step': step, 'image_id': row['image_id'], 'missing_gt': annotation.missing,
                                     'stop_credit': annotation.stops,
                                     'box_credit': [{'label': p.label, 'gt_index': p.gt_index,
                                                     'iou': p.quality, 'credit': p.credit}
                                                    for p in annotation.predictions]})
    report['examples'] = examples
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
