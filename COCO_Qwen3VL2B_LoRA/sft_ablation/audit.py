"""Audit actual COCO targets and real image/token batches before costly SFT."""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor

import baseline_train
from data import load_image, user_message
from train import Collator


def groups(answer):
    return dict(re.findall(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>(.*?)<\|box_end\|>', answer))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    args = parser.parse_args(); root = Path(args.root).resolve()
    folder = root / 'sft_ablation/prepared'; config = json.loads((root / 'sft_ablation/config.json').read_text())
    original = [json.loads(line) for line in (root / config['train_rows']).open()]
    by_id = {r['image_id']: r for r in original}
    result = {'original_images': len(original), 'methods': {}}
    for method in ('m34','m36','m37','m38'):
        rows = [json.loads(line) for line in (folder / f'train_{method}.jsonl').open()]
        if len(rows) != len(original):raise ValueError('Unequal training update budget')
        for row in rows:
            source = by_id[row['image_id']]
            if method in ('m36','m37') and row != source:
                raise ValueError('An unchanged image lost its original prompt or complete target')
            if method == 'm34':
                old, new = groups(source['answer']), groups(row['answer'])
                if set(new) != set(row['query_categories']):raise ValueError('Subset query/answer label mismatch')
                if any(new[k] != old[k] for k in new):raise ValueError('A queried category lost a GT instance')
                if row['full_query'] and (row['prompt'] != source['prompt'] or row['answer'] != source['answer']):
                    raise ValueError('Full query deviated from original benchmark target')
            if method == 'm38':
                old, new = groups(source['answer']), groups(row['answer'])
                if set(old) != set(new):raise ValueError('Augmentation lost a category')
                if any(len(re.findall(r'<\d+>',old[k])) != len(re.findall(r'<\d+>',new[k])) for k in old):
                    raise ValueError('Augmentation lost an instance')
        result['methods'][method] = {'records_verified': len(rows)}
        del rows
    processor = AutoProcessor.from_pretrained(root / config['tokenizer'],local_files_only=True)
    processor.image_processor.max_pixels = 1024*28*28; processor.image_processor.min_pixels = 16*28*28
    processor.image_processor.size['longest_edge'] = 1024*28*28
    processor.image_processor.size['shortest_edge'] = 16*28*28
    examples = original[:4]
    old = baseline_train.CocoRexCollator(processor,4096)(examples)
    new = Collator(processor,4096,'image_first')(examples)
    if not all(torch.equal(old[k],new[k]) for k in old):
        raise ValueError('Unmodified collator differs from the original on actual images')
    result['unmodified_collator_real_batch_equal'] = True
    for method in ('m34','m36','m37','m38'):
        rows = [json.loads(line) for line in (folder / f'smoke_train_{method}.jsonl').open()]
        order = 'text_first' if method == 'm37' else 'image_first'
        batch = Collator(processor,4096,order)(rows[:4])
        text = processor.apply_chat_template(user_message('ORDER_SENTINEL',order=order),tokenize=False,add_generation_prompt=True)
        before = text.index('ORDER_SENTINEL') < text.index('<|vision_start|>')
        if before != (order == 'text_first'):raise ValueError('Chat template changed requested modality order')
        result['methods'][method].update(actual_input_order=order,real_batch_shape=list(batch['input_ids'].shape),
                                          supervised_tokens=int((batch['labels']!=-100).sum()))
        del batch
    example = json.loads((folder/'smoke_train_m38.jsonl').open().readline())
    original_image = load_image(by_id[example['image_id']]); augmented = load_image(example)
    panel = Image.new('RGB',(original_image.width*2,original_image.height),'white')
    panel.paste(original_image,(0,0));panel.paste(augmented,(original_image.width,0))
    # Draw the actual serialized GT on both images for visual inspection.
    for x_offset,record in ((0,by_id[example['image_id']]),(original_image.width,example)):
        draw=ImageDraw.Draw(panel)
        for payload in groups(record['answer']).values():
            coords=[int(v) for v in re.findall(r'<(\d+)>',payload)]
            for pos in range(0,len(coords),4):
                x1,y1,x2,y2=coords[pos:pos+4]
                draw.rectangle((x_offset+x1/999*original_image.width,y1/999*original_image.height,
                                x_offset+x2/999*original_image.width,y2/999*original_image.height),outline='red',width=2)
    panel.save(folder/'m38_geometry_audit.png');original_image.close();augmented.close();panel.close()
    (folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('REAL_COCO_DATA_AND_TOKEN_BATCH_AUDIT_PASSED',json.dumps(result),flush=True)


if __name__ == '__main__':main()
