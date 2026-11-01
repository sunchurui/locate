"""Shared frozen-model inference, raw-token capture and COCO-compatible parsing."""
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, LogitsProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PROMPT = ('Detect all objects from the 80 MS COCO categories. Return each category with its boxes '
          'using the object and box markers and coordinate tokens. Return None if no objects are present.')
FORMAT = (' Return each category with its boxes using <|object_ref_start|>category<|object_ref_end|>'
          '<|box_start|><x1><y1><x2><y2><|box_end|> and coordinate tokens <0> through <999>. '
          'Return None if no objects are present. Do not include explanations.')
TARGET = 'objects belonging to the 80 MS COCO categories'
FIRST, LAST = 150643, 151642


def stable_seed(seed, *parts):
    return int.from_bytes(hashlib.sha256((':'.join(map(str, (seed, *parts)))).encode()).digest()[:4], 'big')


def iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1e-12, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def parse(result, tokenizer, width, height):
    ids = result['token_ids']
    probs = result['coordinate_probabilities']
    marker = {name: tokenizer.convert_tokens_to_ids(name) for name in
              ('<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>')}
    output = []
    position = 0
    while position < len(ids):
        if ids[position] != marker['<|object_ref_start|>']:
            position += 1
            continue
        try:
            end_label = ids.index(marker['<|object_ref_end|>'], position + 1)
            start_box = ids.index(marker['<|box_start|>'], end_label + 1)
            end_box = ids.index(marker['<|box_end|>'], start_box + 1)
        except ValueError:
            break
        name = tokenizer.decode(ids[position + 1:end_label], skip_special_tokens=False).strip().lower()
        coordinates = [i for i in range(start_box + 1, end_box) if FIRST <= ids[i] <= LAST]
        for index in range(0, len(coordinates) - 3, 4):
            positions = coordinates[index:index + 4]
            q = [ids[i] - FIRST for i in positions]
            x1, x2 = sorted((q[0] * width / 999, q[2] * width / 999))
            y1, y2 = sorted((q[1] * height / 999, q[3] * height / 999))
            if x2 > x1 and y2 > y1:
                output.append({'label': name, 'box': [x1, y1, x2, y2],
                               'confidence': sum(probs[i] for i in positions) / 4,
                               'coordinate_positions': positions})
        position = end_box + 1
    return output


def detections(boxes, categories, confidence=False):
    result = []
    for rank, pred in enumerate(boxes):
        if pred['label'] not in categories:
            continue
        x1, y1, x2, y2 = pred['box']
        result.append({'category_id': categories[pred['label']], 'bbox': [x1, y1, x2 - x1, y2 - y1],
                       'score': max(1e-9, min(1., pred['confidence'])) if confidence else max(.001, 1 - rank * .001)})
    return result


class CoordinateRecorder(LogitsProcessor):
    """Record only the chosen coordinate probabilities, never full-vocab histories.

    Appended AFTER upstream temperature/top-k/top-p processors, matching CANL's
    softmax(generate_output.scores) convention. Returns logits unchanged.
    """
    def __init__(self):
        self.pending = None
        self.prefix_length = None
        self.history = []

    def flush(self, ids):
        if self.pending is None:
            return
        if ids.shape[1] != self.prefix_length + 1:
            raise ValueError('Generation token/probability alignment changed')
        token = ids[:, -1]
        valid = (token >= FIRST) & (token <= LAST)
        index = (token - FIRST).clamp(0, LAST - FIRST)
        value = self.pending.gather(1, index[:, None]).squeeze(1) * valid
        self.history.append(value.cpu())

    def __call__(self, input_ids, scores):
        self.flush(input_ids)
        self.prefix_length = input_ids.shape[1]
        # Compute in FP32 like the author's softmax, retain only 1000 bins.
        values = scores.float()
        normalizer = torch.logsumexp(values, dim=-1, keepdim=True)
        self.pending = torch.exp(values[:, FIRST:LAST + 1] - normalizer)
        return scores


class Engine:
    def __init__(self, root, config, output):
        self.root, self.config, self.output = Path(root), config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.processor = AutoProcessor.from_pretrained(self.root / config['adapter'], local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        self.tokenizer = self.processor.tokenizer
        if [self.tokenizer.convert_tokens_to_ids(f'<{q}>') for q in range(1000)] != list(range(FIRST, LAST + 1)):
            raise ValueError('Wrong coordinate-ID tokenizer')
        # Preserve the established BF16 baseline: V100 FP16 and FP32 both
        # degenerated on real image463522 (1024 tokens vs BF16's normal109).
        if torch.cuda.get_device_capability()[0] < 8:
            raise ValueError('Use an Ampere/newer BF16 GPU for comparable Rex-ID inference')
        dtype = torch.bfloat16
        base = Qwen3VLForConditionalGeneration.from_pretrained(self.root / config['model'], torch_dtype=dtype,
                                                              device_map='cuda', local_files_only=True)
        self.model = PeftModel.from_pretrained(base, self.root / config['adapter']).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.cache = {}
        self.accessed = []
        self.calls, self.generated_tokens, self.wall_seconds = 0, 0, 0.
        self.recorder_verified = False
        self.log_path = self.output / 'raw_generations.jsonl'
        if self.log_path.exists():
            for line in self.log_path.open():
                row = json.loads(line)
                self.cache[row['key']] = row['result']
        self.metadata = {'torch': torch.__version__, 'dtype': str(dtype), 'gpu': torch.cuda.get_device_name(),
                         'adapter': str((self.root / config['adapter']).resolve()),
                         'generation_batch': config['generation_batch']}

    def generate(self, image, prompt, key, samples=1, temperature=0., top_k=0, top_p=1., max_tokens=None):
        digest = hashlib.sha256(json.dumps([key, prompt, samples, temperature, top_k, top_p, max_tokens],
                                           sort_keys=True).encode()).hexdigest()
        self.accessed.append(digest)
        if digest in self.cache:
            return self.cache[digest]
        remaining, results, offset = samples, [], 0
        batch_size = min(samples, self.config['generation_batch'])
        while remaining:
            count = min(batch_size, remaining)
            seed = stable_seed(self.config['seed'], key, offset)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            messages = [[{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                      {'type': 'text', 'text': prompt}]}] for _ in range(count)]
            texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
            images, videos = process_vision_info(messages)
            inputs = self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors='pt').to('cuda')
            recorder = CoordinateRecorder()
            # PEFT delegates generate to this underlying Qwen model.
            core = self.model.get_base_model()
            original = core._get_logits_processor

            def get_processors(*args, **kwargs):
                processors = original(*args, **kwargs)
                processors.append(recorder)
                return processors

            core._get_logits_processor = get_processors
            started = time.perf_counter()
            try:
                kwargs = {'max_new_tokens': max_tokens or self.config['max_new_tokens'],
                          'do_sample': temperature > 0, 'use_cache': True}
                if temperature > 0:
                    kwargs.update(temperature=temperature, top_k=top_k, top_p=top_p)
                verify = not self.recorder_verified and count == 1
                if verify:
                    kwargs.update(return_dict_in_generate=True, output_scores=True)
                with torch.inference_mode():
                    returned = self.model.generate(**inputs, **kwargs)
                generated = returned.sequences if verify else returned
                recorder.flush(generated)
                ids = generated[:, inputs.input_ids.shape[1]:].cpu().tolist()
                probabilities = torch.stack(recorder.history, dim=1).tolist()
                if any(len(a) != len(b) for a, b in zip(ids, probabilities)):
                    raise ValueError('Coordinate confidence history is misaligned')
                if verify:
                    checked = 0
                    for position, scores in enumerate(returned.scores):
                        token = ids[0][position]
                        if FIRST <= token <= LAST:
                            expected = torch.softmax(scores[0].float(), dim=-1)[token].item()
                            if abs(expected - probabilities[0][position]) > 2e-6:
                                raise ValueError('Streaming coordinate probability differs from official score softmax')
                            checked += 1
                    print(f'COORDINATE_PROBABILITY_VERIFIED: {checked} actual coordinate tokens', flush=True)
                    self.recorder_verified = True
            except torch.cuda.OutOfMemoryError:
                if count == 1:
                    raise
                batch_size = max(1, count // 2)
                del inputs, messages, images, videos, recorder
                torch.cuda.empty_cache()
                print(f'OOM: retry generation batch {batch_size}', flush=True)
                continue
            finally:
                core._get_logits_processor = original
            elapsed = time.perf_counter() - started
            self.calls += count
            self.wall_seconds += elapsed
            for tokens, probs in zip(ids, probabilities):
                if self.tokenizer.eos_token_id in tokens:
                    end = tokens.index(self.tokenizer.eos_token_id) + 1
                    tokens, probs = tokens[:end], probs[:end]
                self.generated_tokens += len(tokens)
                results.append({'token_ids': tokens, 'coordinate_probabilities': probs,
                                'text': self.tokenizer.decode(tokens, skip_special_tokens=False,
                                                              clean_up_tokenization_spaces=False),
                                'actual_batch': count, 'seed': seed,
                                'generation_seconds': elapsed / count,
                                'truncated': not tokens or tokens[-1] != self.tokenizer.eos_token_id})
            remaining -= count; offset += count
            del inputs, generated, returned, messages, images, videos, recorder
        self.cache[digest] = results
        with self.log_path.open('a') as f:
            f.write(json.dumps({'key': digest, 'request_key': key, 'prompt': prompt, 'result': results}) + '\n')
        return results

    def cost(self, keys):
        responses = [item for key in set(keys) for item in self.cache[key]]
        return {'responses': len(responses), 'tokens': sum(len(item['token_ids']) for item in responses),
                'generation_seconds': sum(item['generation_seconds'] for item in responses),
                'generate_batches': round(sum(1 / item['actual_batch'] for item in responses))}
