"""COCO adaptations of Rex query sampling and Phi-Ground Algorithms 2/3."""
import hashlib
import importlib.util
import json
import random
from pathlib import Path

from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rng_for(seed, method, image_id):
    value = hashlib.sha256(f'{seed}:{method}:{image_id}'.encode()).digest()
    return random.Random(int.from_bytes(value[:8], 'big'))


def formatter(root, expected_commit):
    import subprocess
    source = Path(root) / 'references/Rex-Omni'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if commit != expected_commit:
        raise ValueError(f'Unexpected Rex source revision: {commit}')
    path = source / 'finetuning/dataset/task_fns/grounding_task.py'
    # Read the immutable git blob, rather than trusting possible worktree edits.
    raw = subprocess.check_output(['git', 'show', f'{commit}:finetuning/dataset/task_fns/grounding_task.py'], cwd=source)
    scope = {'__name__': 'pinned_rex_grounding'}
    exec(compile(raw, str(path), 'exec'), scope)
    return scope['GroundingTaskFn'](task_prompts=[], image_min_pixels=0, image_max_pixels=0), hashlib.sha256(raw).hexdigest()


def serialize(task, boxes, width, height):
    qa = {name: task.convert_boxes_from_absolute_to_normalized_bins(boxes[name], width, height)
          for name in sorted(boxes)}
    return task.compose_answer(qa) if qa else 'None'


def resize_plan(width, height, settings, rng):
    # Phi-Ground Algorithm2. COCO adaptation: target canvas retains input W,H.
    max_scale = 1.0
    min_scale = min(1.0, width / settings['max_screen_size']) * max_scale
    scale = rng.uniform(min_scale, max_scale)
    resized_w = max(1, min(width, int(width * scale)))
    resized_h = max(1, min(height, int(height * scale)))
    return {'canvas': [width, height], 'resized': [resized_w, resized_h],
            'offset': [rng.randint(0, width - resized_w), rng.randint(0, height - resized_h)],
            'background': settings['background'], 'resample': settings['resample']}


def transform_boxes(boxes, plan):
    width, height = plan['canvas']
    new_w, new_h = plan['resized']
    dx, dy = plan['offset']
    return {name: [[x1 * new_w / width + dx, y1 * new_h / height + dy,
                    x2 * new_w / width + dx, y2 * new_h / height + dy]
                   for x1, y1, x2, y2 in values] for name, values in boxes.items()}


def load_image(example):
    with Image.open(example['image']) as source:
        image = source.convert('RGB')
    plan = example.get('augmentation')
    if plan:
        if list(image.size) != plan['canvas']:
            raise ValueError('Image dimensions differ from the frozen augmentation manifest')
        resized = image.resize(tuple(plan['resized']), Image.Resampling.BICUBIC)
        canvas = Image.new('RGB', tuple(plan['canvas']), tuple(plan['background']))
        canvas.paste(resized, tuple(plan['offset']))
        image.close(); resized.close()
        image = canvas
    return image


def user_message(prompt, image=None, order='image_first'):
    visual = {'type': 'image'}
    if image is not None:
        visual['image'] = image
    text = {'type': 'text', 'text': prompt}
    content = [text, visual] if order == 'text_first' else [visual, text]
    return [{'role': 'user', 'content': content}]
