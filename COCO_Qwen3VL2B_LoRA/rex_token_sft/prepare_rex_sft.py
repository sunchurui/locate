#!/usr/bin/env python3
"""Create COCO SFT examples using the official Rex-Omni answer serializer."""
import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

PROMPT = ('Detect all objects from the 80 MS COCO categories. Return each category with its boxes '
          'using the object and box markers and coordinate tokens. Return None if no objects are present.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rex-source', required=True)
    args = parser.parse_args()
    task_path = Path(args.rex_source) / 'finetuning/dataset/task_fns/grounding_task.py'
    spec = importlib.util.spec_from_file_location('rex_grounding_task', task_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    formatter = module.GroundingTaskFn(task_prompts=[], image_min_pixels=0, image_max_pixels=0)
    with open(args.instances, encoding='utf-8') as f:
        data = json.load(f)
    cats = {item['id']: item['name'] for item in data['categories']}
    annotations = defaultdict(lambda: defaultdict(list))
    for ann in data['annotations']:
        if ann.get('iscrowd', 0):
            continue
        x, y, w, h = ann['bbox']
        if w > 0 and h > 0:
            annotations[ann['image_id']][cats[ann['category_id']]].append([x, y, x + w, y + h])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir).resolve()
    count = 0
    with output.open('w', encoding='utf-8', newline='\n') as out:
        for info in sorted(data['images'], key=lambda x: x['id']):
            boxes = annotations[info['id']]
            qa = {name: formatter.convert_boxes_from_absolute_to_normalized_bins(
                boxes[name], info['width'], info['height']) for name in sorted(boxes)}
            answer = formatter.compose_answer(qa) if qa else 'None'
            row = {'image_id': info['id'], 'image': str(image_dir / info['file_name']),
                   'prompt': PROMPT, 'answer': answer}
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    print(f'{output}: {count} images', flush=True)


if __name__ == '__main__':
    main()
