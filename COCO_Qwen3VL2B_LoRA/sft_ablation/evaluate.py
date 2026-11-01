"""Same Rex-ID parser/scoring as baseline; supports matched text-first evaluation."""
import argparse
import contextlib
import gc
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from qwen_vl_utils import process_vision_info

from data import sha256, user_message

from baseline_evaluate import PROMPT, parse_output

if sha256(Path(__file__).with_name('baseline_evaluate.py')) != 'c93882eecfbf78bcd3df9dbfbd77f7b0f2dca445b5648553ff22135617f2ab8e':
    raise ValueError('The frozen baseline Rex-ID parser/evaluator has changed')


def to_detections(response, info, categories):
    dets = []
    for rank, item in enumerate(parse_output(response)):
        label = str(item.get('label', '')).strip().lower()
        if label not in categories:
            continue
        x1, y1, x2, y2 = [float(v) for v in item['bbox_2d']]
        x1, x2 = sorted((max(0., min(1000., x1)), max(0., min(1000., x2))))
        y1, y2 = sorted((max(0., min(1000., y1)), max(0., min(1000., y2))))
        w, h = (x2 - x1) * info['width'] / 1000, (y2 - y1) * info['height'] / 1000
        if w > 0 and h > 0:
            dets.append({'category_id': categories[label], 'bbox': [x1 * info['width'] / 1000,
                         y1 * info['height'] / 1000, w, h], 'score': max(.001, 1 - rank * .001)})
    return dets


def generate_batch(model, processor, root, images, order, max_tokens):
    conversations = [user_message(PROMPT, str(root / 'data/coco2017/val2017' / info['file_name']), order)
                     for info in images]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in conversations]
    visuals, videos = process_vision_info(conversations)
    inputs = processor(text=texts, images=visuals, videos=videos, padding=True, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
    tokens = generated[:, inputs.input_ids.shape[1]:]
    responses = processor.batch_decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return responses


def run(model, processor, root, output, order, batch_size=16, max_tokens=1024, manifest=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    annotation = root / 'data/coco2017/annotations/instances_val2017.json'
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
    wanted = sorted(gt.imgs)
    if manifest:
        data = json.loads(Path(manifest).read_text())
        if sha256(annotation) != data['annotation_sha256']:
            raise ValueError('Progress validation manifest annotation mismatch')
        wanted = sorted(data['image_ids'])
    predictions = output / 'predictions.jsonl'
    recorded = {}
    if predictions.exists():
        for line in predictions.open():
            row = json.loads(line); recorded[row['image_id']] = row
    if not set(recorded).issubset(wanted):
        raise ValueError('Evaluation output contains images outside this evaluation')
    categories = {item['name'].lower(): item['id'] for item in gt.dataset['categories']}
    pending = [gt.imgs[i] for i in wanted if i not in recorded]
    was_training, padding = model.training, processor.tokenizer.padding_side
    old_cache = model.config.use_cache
    processor.tokenizer.padding_side = 'left'; model.eval()
    started = time.perf_counter()
    try:
        offset = 0
        with predictions.open('a') as stream:
            while offset < len(pending):
                batch = pending[offset:offset + batch_size]
                try:
                    responses = generate_batch(model, processor, root, batch, order, max_tokens)
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2); gc.collect(); torch.cuda.empty_cache()
                    print('Progress evaluation retries with batch', batch_size, flush=True)
                    continue
                for info, response in zip(batch, responses):
                    row = {'image_id': info['id'], 'raw_output': response,
                           'detections': to_detections(response, info, categories)}
                    stream.write(json.dumps(row) + '\n'); recorded[info['id']] = row
                stream.flush(); offset += len(batch)
                if offset % 100 < batch_size or offset == len(pending):
                    print(f'AP evaluation {len(recorded)}/{len(wanted)}', flush=True)
        if set(recorded) != set(wanted):
            raise ValueError('Incomplete evaluation coverage')
        detections = [dict(d, image_id=i) for i in wanted for d in recorded[i]['detections']]
        with contextlib.redirect_stdout(io.StringIO()):
            if detections:
                dt = gt.loadRes(detections)
            else:
                dt = COCO(); dt.dataset = {'images': [gt.imgs[i] for i in wanted],
                                          'categories': gt.dataset['categories'], 'annotations': []}; dt.createIndex()
            evaluator = COCOeval(gt, dt, 'bbox'); evaluator.params.imgIds = wanted
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        names = ['AP@[.50:.95]', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR@1', 'AR@10', 'AR@100']
        metrics = {name: float(v) for name, v in zip(names, evaluator.stats)}
        metrics.update(images=len(wanted), detections=len(detections), input_order=order,
                       seconds=time.perf_counter() - started, batch_size=batch_size,
                       score_note='Same generated-order scores and q/999 geometry as original Rex-ID SFT evaluation')
        (output / 'coco_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print('COCO_AP_RESULT', json.dumps(metrics), flush=True)
        return metrics
    finally:
        processor.tokenizer.padding_side = padding; model.config.use_cache = old_cache
        model.train(was_training); gc.collect(); torch.cuda.empty_cache()


def main():
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--adapter', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--manifest'); parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args(); root = Path(args.root).resolve(); adapter = Path(args.adapter)
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError('Use a BF16-capable Ampere/newer GPU; V100 pilots were excluded')
    config = json.loads((root / 'sft_ablation/config.json').read_text())
    metadata = json.loads((adapter / 'sft_ablation.json').read_text())
    processor = AutoProcessor.from_pretrained(adapter, local_files_only=True)
    if [processor.tokenizer.convert_tokens_to_ids(f'<{i}>') for i in range(1000)] != list(range(150643,151643)):
        raise ValueError('Expected original Rex coordinate IDs')
    model = Qwen3VLForConditionalGeneration.from_pretrained(root / config['model'], torch_dtype=torch.bfloat16,
                                                          device_map='cuda', local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter).eval()
    run(model, processor, root, Path(args.output), metadata['input_order'], args.batch_size,
        config['train']['max_new_tokens'], args.manifest)


if __name__ == '__main__':
    main()
