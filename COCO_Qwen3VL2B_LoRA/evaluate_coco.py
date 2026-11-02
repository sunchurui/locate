#!/usr/bin/env python3
import argparse, json, os, re, sys, time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pycocotools.coco import COCO
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'unranked_metrics.py').exists())))
from unranked_metrics import evaluate as evaluate_set, load_predictions, save_metrics

PROMPT = "Detect all objects from the 80 MS COCO categories. Return only a JSON array; each item must contain label and bbox_2d [x1,y1,x2,y2], with coordinates normalized to 0-1000."

# Qwen3-VL emits a plain-text JSON array: its boxes carry no confidence. Rather than
# inventing a ranking from output position (which encodes x1-order / box count, not
# reliability), every detection gets the same score, so the PR curve collapses to a
# single operating point. Same convention as Rex-Omni (conf=0.2 for all boxes).
DET_SCORE = 1.0

def parse_output(text):
    try:
        obj = json.loads(text)
    except Exception:
        try:
            from json_repair import loads
            obj = loads(text)
        except Exception:
            m = re.search(r"\[.*\]", text, flags=re.S)
            if not m:
                return []
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return []
    return obj if isinstance(obj, list) else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prediction-file", default="predictions.jsonl")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--skip-coco-eval", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pred_file = Path(args.output_dir) / args.prediction_file
    metrics_file = Path(args.output_dir) / "set_metrics.json"

    coco_gt = COCO(args.annotations)
    cats = {c["name"].lower(): c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    images = sorted(coco_gt.imgs.values(), key=lambda x: x["id"])
    if args.limit:
        images = images[:args.limit]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error("shard-index must be in [0, num-shards)")
    images = images[args.shard_index::args.num_shards]

    if not args.skip_inference:
        if args.batch_size < 1:
            ap.error("batch-size must be positive")
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        processor.tokenizer.padding_side = "left"
        capability = torch.cuda.get_device_capability()
        inference_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=inference_dtype, device_map="cuda", trust_remote_code=True
        )
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

        done = set()
        if pred_file.exists():
            with pred_file.open(encoding="utf-8") as f:
                for line in f:
                    try: done.add(json.loads(line)["image_id"])
                    except Exception: pass

        pending = [info for info in images if info["id"] not in done]
        started = time.time()
        with pred_file.open("a", encoding="utf-8") as out:
            offset = 0
            batch_size = args.batch_size
            while offset < len(pending):
                batch = pending[offset:offset + batch_size]
                conversations = texts = image_inputs = video_inputs = inputs = None
                try:
                    conversations = [[{"role": "user", "content": [
                        {"type": "image", "image": os.path.join(args.image_dir, info["file_name"])},
                        {"type": "text", "text": PROMPT},
                    ]}] for info in batch]
                    texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                             for messages in conversations]
                    image_inputs, video_inputs = process_vision_info(conversations)
                    inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                                       padding=True, return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                                   do_sample=False, use_cache=True)
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    del inputs, conversations, texts, image_inputs, video_inputs
                    torch.cuda.empty_cache()
                    print(f"CUDA memory limit reached; retrying with batch_size={batch_size}", flush=True)
                    continue

                prompt_length = inputs.input_ids.shape[1]
                trimmed = generated[:, prompt_length:]
                responses = processor.batch_decode(trimmed, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
                for info, response in zip(batch, responses):
                    boxes = parse_output(response)
                    dets = []
                    width, height = info["width"], info["height"]
                    for item in boxes:
                        if not isinstance(item, dict):
                            continue
                        label = str(item.get("label", "")).strip().lower()
                        bbox = item.get("bbox_2d")
                        if label not in cats or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                            continue
                        try:
                            x1, y1, x2, y2 = [float(v) for v in bbox]
                        except Exception:
                            continue
                        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
                        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
                        bw, bh = (x2-x1)*width/1000.0, (y2-y1)*height/1000.0
                        if bw <= 0 or bh <= 0:
                            continue
                        dets.append({"category_id": cats[label],
                                     "bbox": [x1*width/1000.0, y1*height/1000.0, bw, bh],
                                     "score": DET_SCORE})
                    out.write(json.dumps({"image_id": info["id"], "detections": dets}) + "\n")
                out.flush()
                offset += len(batch)
                if offset % 100 < batch_size or offset == len(pending):
                    print(f"processed {len(done) + offset}/{len(images)} images; "
                          f"elapsed {time.time()-started:.0f}s; batch_size={batch_size}", flush=True)

    if args.skip_coco_eval:
        return

    recorded = load_predictions(pred_file)
    wanted = [info['id'] for info in images]
    if not set(wanted).issubset(recorded):
        raise ValueError('Incomplete prediction coverage; no final metrics written')
    metrics = evaluate_set(coco_gt, recorded, wanted)
    save_metrics(args.output_dir, metrics)
    print('UNRANKED_SET_METRICS', json.dumps(metrics), flush=True)

if __name__ == "__main__":
    main()
