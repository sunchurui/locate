"""Existing Swift GRPO with common epoch3 SFT initialization and progress eval."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import torch
import torch.distributed.fsdp as fsdp
if not hasattr(fsdp, 'FSDPModule'):
    fsdp.FSDPModule = fsdp.FullyShardedDataParallel
from transformers import TrainerCallback
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

root = Path(os.environ['EXPERIMENT_ROOT']).resolve()
adapter = Path(os.environ['SFT_ADAPTER']).resolve()
output = Path(os.environ['OUTPUT_DIR']).resolve()
variant = os.environ['RL_VARIANT']
assert adapter == root/'rex_id_sft_outputs_5epoch_b64/checkpoint-5547'
state = json.loads((adapter/'trainer_state.json').read_text())
assert state['global_step'] == 5547 and state['epoch'] == 3
if output.exists() and any(output.iterdir()):
    raise RuntimeError('Fresh paired RL requires an empty output; preserve old runs')
for name in ['tokenizer.json','tokenizer_config.json','preprocessor_config.json','chat_template.jinja']:
    if (adapter/name).read_bytes() != (root/'rex_id_rl_base'/name).read_bytes():
        raise RuntimeError(f'SFT and RL metadata differ: {name}')
if (root/'rex_id_rl_base/adapter_model.safetensors').exists():
    raise RuntimeError('Base overlay unexpectedly contains adapter weights')
assert (variant == 'credit_norm') == bool(os.environ.get('OBJECT_CREDIT_CONFIG'))
output.mkdir(parents=True, exist_ok=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

record={'variant':variant,'policy_adapter':str(adapter),'frozen_reference_adapter':str(adapter),
        'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
        'train_data_sha256':sha(root/'data/coco2017/annotations/train_rex_id_grpo_epoch.jsonl'),
        'val_manifest_sha256':sha(root/'rl_val_manifest.json'),
        'epochs':1,'optimizer_updates':313,'seed':20260812,'learning_rate':5e-7,
        'distinct_prompts_per_update':32,'responses_per_prompt':8,
        'responses_per_update':256,'microbatch':32,'gradient_accumulation':8,
        'progress_evaluation_steps':[100,200,300],
        'evaluation':'Global endpoint micro P/R/F1 with Rex matching; all-query outputs'}
snapshot=output/'source_snapshot';snapshot.mkdir()
for directory in ['rex_id_epoch3_rl','rex_id_object_credit','rex_id_grpo']:
    target=snapshot/directory;target.mkdir()
    for source in (root/directory).iterdir():
        if source.is_file():shutil.copy2(source,target/source.name)
for name in ['unranked_metrics.py','official_metrics.py','micro_metrics.py','rex_token_sft/evaluate_rex_id_coco.py']:
    target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/name,target)
(output/'paired_experiment.json').write_text(json.dumps(record,indent=2)+'\n')
print('EPOCH3_PAIRED_EXPERIMENT',json.dumps(record),flush=True)

class ProgressEval(TrainerCallback):
    def on_save(self,args,state,control,**kwargs):
        if state.global_step not in (100,200,300) or not state.is_world_process_zero:
            return control
        checkpoint=output/f'checkpoint-{state.global_step}'
        for name in ['tokenizer.json','tokenizer_config.json','preprocessor_config.json',
                     'video_preprocessor_config.json','chat_template.jinja','special_tokens_map.json']:
            if (adapter/name).exists():shutil.copy2(adapter/name,checkpoint/name)
        target=root/'evaluation/epoch3_rl'/variant/f'step{state.global_step}_val256'
        target.mkdir(parents=True,exist_ok=True)
        torch.cuda.empty_cache()
        cmd=[sys.executable,str(root/'rex_token_sft/evaluate_rex_id_coco.py'),
             '--model',str(root/'model'),'--adapter',str(checkpoint),
             '--annotations',str(root/'data/coco2017/annotations/instances_val2017.json'),
             '--image-dir',str(root/'data/coco2017/val2017'),
             '--image-ids',str(root/'rl_val_manifest.json'),
             '--output-dir',str(target),'--batch-size','8']
        print('PROGRESS_EVAL_START',state.global_step,str(target),flush=True)
        try:
            result=subprocess.run(cmd,timeout=1800)
            status={'returncode':result.returncode}
        except subprocess.TimeoutExpired:
            status={'returncode':None,'error':'progress evaluation timed out after1800 seconds'}
        status.update(step=state.global_step,metrics=str(target/'micro_metrics.json'))
        (target/'evaluation_status.json').write_text(json.dumps(status,indent=2)+'\n')
        print('PROGRESS_EVAL_END',json.dumps(status),flush=True)
        return control

original_init=GRPOTrainer.__init__
def initialize(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    self.add_callback(ProgressEval())
GRPOTrainer.__init__=initialize
runpy.run_path(str(root/'rex_id_grpo/run_rlhf.py'),run_name='__main__')
