"""Locate the final completed Swift checkpoint and supply its tokenizer metadata."""
import json
import shutil
import sys
from pathlib import Path

output,overlay=map(Path,sys.argv[1:])
candidates=[p for p in output.glob('checkpoint-*')
            if p.name.split('-')[-1].isdigit() and (p/'adapter_config.json').exists()]
if not candidates:
    raise FileNotFoundError(f'No RL adapter checkpoints in {output}')
checkpoint=max(candidates,key=lambda p:int(p.name.split('-')[-1]))
state=json.loads((checkpoint/'trainer_state.json').read_text())
if state['global_step'] < state['max_steps']:
    raise ValueError('RL checkpoint is incomplete; resume training before final COCO evaluation')
config=json.loads((checkpoint/'adapter_config.json').read_text())
if config.get('trainable_token_indices') != list(range(150643,151643)):
    raise ValueError('RL checkpoint lost the coordinate token-row adapter configuration')
for name in ('tokenizer.json','tokenizer_config.json','preprocessor_config.json',
             'video_preprocessor_config.json','chat_template.json','chat_template.jinja',
             'processor_config.json','special_tokens_map.json'):
    source=overlay/name
    if source.exists() and not (checkpoint/name).exists():
        shutil.copy2(source,checkpoint/name)
print(checkpoint.resolve())
