#!/usr/bin/env python3
"""Reuse fixed IDs and the SFT tokenizer without copying model weight shards."""
import argparse
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
                 'video_preprocessor_config.json','chat_template.json','special_tokens_map.json'):
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
    print('Base overlay:',overlay,flush=True)


if __name__=='__main__':
    main()
