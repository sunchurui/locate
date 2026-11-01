"""Full COCO evaluation after completion or user cancellation; preserve all checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(); p.add_argument('--root', required=True); args = p.parse_args()
root = Path(args.root).resolve(); output = root / 'rex_id_sft_outputs_5epoch_b64'
checkpoints = sorted(output.glob('checkpoint-*'), key=lambda x: int(x.name.split('-')[-1]), reverse=True)
adapter = None
for candidate in checkpoints:
    if all((candidate / name).exists() for name in ('adapter_model.safetensors', 'trainer_state.json', 'chat_template.jinja')):
        adapter = candidate; break
if adapter is None:
    raise RuntimeError('No complete checkpoint exists yet; nothing will be silently substituted')
state = json.loads((adapter / 'trainer_state.json').read_text())
target = root / 'evaluation/rex_id_sft_5epoch_b64_full' / adapter.name
print('FULL_EVALUATION', adapter, 'epoch', state.get('epoch'), flush=True)
subprocess.run([sys.executable, str(root / 'rex_token_sft/evaluate_rex_id_coco.py'),
 '--model', str(root / 'model'), '--adapter', str(adapter),
 '--annotations', str(root / 'data/coco2017/annotations/instances_val2017.json'),
 '--image-dir', str(root / 'data/coco2017/val2017'), '--output-dir', str(target), '--batch-size', '16'], check=True)
