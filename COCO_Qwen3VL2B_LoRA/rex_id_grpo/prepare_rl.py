#!/usr/bin/env python3
"""Reuse fixed IDs and the SFT tokenizer without copying model weight shards."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROMPT = ('<image>Detect all objects from the 80 MS COCO categories. Return each category with its boxes '
          'using the object and box markers and coordinate tokens. Return None if no objects are present.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    base, adapter = root/'model', root/'rex_id_sft_outputs_a800_80'
    overlay = root/'rex_id_rl_base'
    overlay.mkdir(exist_ok=True)
    for source in base.glob('*.safetensors*'):
        dest = overlay/source.name
        if not dest.exists():
            dest.symlink_to(source)
    for name in ('config.json', 'generation_config.json'):
        shutil.copy2(base/name, overlay/name)
    for name in ('tokenizer.json','tokenizer_config.json','preprocessor_config.json',
                 'video_preprocessor_config.json','chat_template.json','chat_template.jinja',
                 'processor_config.json','special_tokens_map.json'):
        source = adapter/name
        if source.exists():
            shutil.copy2(source,overlay/name)
    config=json.loads((adapter/'adapter_config.json').read_text())
    if config.get('trainable_token_indices') != list(range(150643,151643)):
        raise ValueError('SFT adapter must train the Rex original coordinate ID range')
    tok=json.loads((overlay/'tokenizer.json').read_text())
    if any(tok['model']['vocab'].get(f'<{i}>') != 150643+i for i in range(1000)):
        raise ValueError('SFT tokenizer does not use the required coordinate IDs')
    data=root/'data/coco2017/annotations'
    for source,manifest,target in [('train_rl_10000.jsonl','rl_train_10000_manifest.json','train_rex_id_grpo_10000.jsonl'),
                                   ('val_rl_256.jsonl','rl_val_manifest.json','val_rex_id_grpo_256.jsonl')]:
        rows=[json.loads(line) for line in (data/source).open()]
        expected=json.loads((root/manifest).read_text())['image_ids']
        if [r['image_id'] for r in rows] != expected:
            raise ValueError(f'{source} differs from the fixed subset manifest')
        with (data/target).open('w',encoding='utf-8') as f:
            for row in rows:
                row['messages']=[{'role':'user','content':PROMPT}]
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(target,len(rows),flush=True)
        if source=='train_rl_10000.jsonl':
            ordered=sorted(rows,key=lambda row:(hashlib.sha256(
                f'coco2017:rl-order:seed=20260812:image_id={row["image_id"]}'.encode()).digest(),row['image_id']))
            padding=(-len(ordered))%32
            ordered=ordered+ordered[:padding]
            if any(len({r['image_id'] for r in ordered[i:i+32]})!=32 for i in range(0,len(ordered),32)):
                raise ValueError('Each update must contain 32 different images')
            with (data/'train_rex_id_grpo_epoch.jsonl').open('w',encoding='utf-8') as f:
                for row in ordered:
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
            order_manifest={'seed':20260812,'unique_images':len(rows),'prompt_draws':len(ordered),
                            'padded_prompts':padding,'optimizer_updates':len(ordered)//32,
                            'selection':'SHA256 coco2017:rl-order:seed=20260812:image_id={id}; append first padding rows',
                            'image_ids':[r['image_id'] for r in ordered]}
            (root/'rex_id_grpo_epoch_manifest.json').write_text(json.dumps(order_manifest,indent=2)+'\n')
            print('Epoch:',len(ordered),'prompts,',len(ordered)//32,'optimizer updates; padded',padding,flush=True)
    print('Base overlay:',overlay,flush=True)


if __name__=='__main__':
    main()
