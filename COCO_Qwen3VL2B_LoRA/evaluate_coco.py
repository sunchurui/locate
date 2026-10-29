#!/usr/bin/env python3
import argparse, json, os, re, sys, time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROMPT = "Detect all objects from the 80 MS COCO categories. Return only a JSON array; each item must contain label and bbox_2d [x1,y1,x2,y2], with coordinates normalized to 0-1000."

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
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pred_file = Path(args.output_dir) / "predictions.jsonl"
    metrics_file = Path(args.output_dir) / "coco_metrics.json"

    coco_gt = COCO(args.annotations)
    cats = {c["name"].lower(): c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    images = sorted(coco_gt.imgs.values(), key=lambda x: x["id"])
    if args.limit:
        images = images[:args.limit]
    done = set()
    if pred_file.exists():
        with pred_file.open(encoding="utf-8") as f:
            for line in f:
                try: done.add(json.loads(line)["image_id"])
                except Exception: pass

    started = time.time()
    with pred_file.open("a", encoding="utf-8") as out:
        for idx, info in enumerate(images, 1):
            image_id = info["id"]
            if image_id in done:
                continue
            path = os.path.join(args.image_dir, info["file_name"])
            messages = [{"role": "user", "content": [
                {"type": "image", "image": path},
                {"type": "text", "text": PROMPT},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, use_cache=True)
            trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, generated)]
            response = processor.batch_decode(trimmed, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
            boxes = parse_output(response)
            dets = []
            width, height = info["width"], info["height"]
            for rank, item in enumerate(boxes):
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
                             "score": max(0.001, 1.0 - rank*0.001)})
            out.write(json.dumps({"image_id": image_id, "detections": dets}) + "\n")
            out.flush()
            if idx % 100 == 0 or idx == len(images):
                print(f"processed {idx}/{len(images)} images; elapsed {time.time()-started:.0f}s", flush=True)

    predictions = []
    with pred_file.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            predictions.extend(dict(d, image_id=row["image_id"]) for d in row["detections"])
    if not predictions:
        metrics = {"images": len(images), "detections": 0}
    else:
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [x["id"] for x in images]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics = {
            "images": len(images), "detections": len(predictions),
            "AP@[.50:.95]": float(evaluator.stats[0]),
            "AP50": float(evaluator.stats[1]),
            "AP75": float(evaluator.stats[2]),
            "AP_small": float(evaluator.stats[3]),
            "AP_medium": float(evaluator.stats[4]),
            "AP_large": float(evaluator.stats[5]),
            "AR@1": float(evaluator.stats[6]),
            "AR@10": float(evaluator.stats[7]),
            "AR@100": float(evaluator.stats[8]),
            "score_note": "Qwen3-VL output has no calibrated confidence target; per-image generated order is used as the detection score ranking."
        }
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
