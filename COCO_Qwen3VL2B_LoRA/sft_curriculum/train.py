"""Paired frozen-order curriculum ablation using the established Rex-ID LoRA trainer."""
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from transformers import TrainerCallback,set_seed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--variant',choices=['curriculum','random_control'],required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root).resolve();folder=root/'sft_curriculum'
    sys.path.insert(0,str(root/'sft_ablation'))
    import baseline_train as baseline
    from data import sha256
    spec=importlib.util.spec_from_file_location('sft_helpers',root/'sft_ablation/train.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    report=json.loads((folder/'prepared/report.json').read_text())
    config=json.loads((root/'sft_ablation/config.json').read_text())
    assert report['inputs']['baseline_train']==config['baseline_trainer_sha256']==sha256(root/'sft_ablation/baseline_train.py')
    assert report['inputs']['val_rows']==sha256(root/config['val_rows'])
    assert report['inputs']['val_manifest']==sha256(root/'rl_val_manifest.json')
    data=folder/'prepared'/f'{args.variant}.jsonl'
    assert sha256(data)==report['files'][data.name]
    assert sha256(folder/'prepared'/f'{args.variant}_ids.json')==report['files'][f'{args.variant}_ids.json']
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<80_000_000_000:
        raise ValueError('All training and smoke updates require >=80GB GPU')
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be empty; preserve all previous runs')
    output.mkdir(parents=True,exist_ok=True)
    settings=config['train']
    metadata={'method':'m30_'+args.variant,'variant':args.variant,'config':config,'input_order':'image_first',
        'curriculum_report':report,'smoke':args.smoke,'train_sha256':sha256(data),
        'initialization':'fresh original Qwen3-VL-2B plus fresh seeded LoRA; no SFT or RL adapter loaded',
        'sampler':'SequentialSampler over frozen order; same unique118287 examples in both runs',
        'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'evaluation_protocol':'rex_matching_global_endpoint_micro_v1'}
    snapshot=output/'source';snapshot.mkdir()
    for path in [folder/'train.py',folder/'prepare.py',folder/'train.sbatch',folder/'evaluate.sbatch',
                 folder/'README.md',root/'micro_metrics.py',root/'EVALUATION_PROTOCOL.md']:
        if path.exists():shutil.copy2(path,snapshot/path.name)
    # Reuse existing generation-validation and processor-saving callbacks unchanged.
    callback=helpers.Protocol(root,metadata,None,args.smoke)

    class StageLog(TrainerCallback):
        def on_step_end(self,training_args,state,control,**kwargs):
            if args.variant=='curriculum' and state.global_step==report['phase1_end_step']:
                print('M30_PHASE2_MIXED_START',state.global_step,flush=True)
            return control

    original=baseline.Trainer
    class FrozenOrderTrainer(original):
        def _get_train_sampler(self,train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def __init__(self,*positional,**kwargs):
            training=kwargs['args'];training.save_total_limit=None
            training.dataloader_num_workers=2
            training.max_steps=2 if args.smoke else report['optimizer_updates']
            if args.smoke:
                training.dataloader_num_workers=0;training.eval_strategy='no'
                training.save_strategy='no';training.logging_steps=1
            callback.processor=kwargs['processing_class']
            kwargs['callbacks']=list(kwargs.pop('callbacks',[]))+[callback,StageLog()]
            digest=hashlib.sha256()
            for name,parameter in kwargs['model'].named_parameters():
                if parameter.requires_grad:
                    digest.update(name.encode());digest.update(parameter.detach().cpu().view(torch.uint8).numpy().tobytes())
            metadata['initial_trainable_parameters_sha256']=digest.hexdigest()
            (output/'sft_ablation.json').write_text(json.dumps(metadata,indent=2)+'\n')
            print('M30_PAIRED_PROTOCOL',json.dumps(metadata),flush=True)
            super().__init__(*positional,**kwargs)
            # Ensure no framework wrapper substituted a random sampler.
            assert isinstance(self._get_train_sampler(),SequentialSampler)

    baseline.Trainer=FrozenOrderTrainer
    baseline.CocoRexCollator=lambda processor,max_length:helpers.Collator(processor,max_length,'image_first')
    set_seed(42)
    sys.argv=[str(root/'sft_ablation/baseline_train.py'),'--model',str(root/config['model']),
        '--tokenizer-dir',str(root/config['tokenizer']),'--train-data',str(data),
        '--val-data',str(root/config['val_rows']),'--output-dir',str(output),
        '--train-batch-size',str(settings['per_device_batch']),'--eval-batch-size','4',
        '--grad-accum',str(settings['gradient_accumulation']),'--epochs','1',
        '--learning-rate',str(settings['learning_rate']),'--eval-steps',str(settings['eval_loss_steps']),
        '--save-steps',str(settings['save_steps']),'--max-length',str(settings['max_length'])]
    baseline.main()
    state=json.loads((output/'trainer_state.json').read_text())
    expected=2 if args.smoke else report['optimizer_updates']
    if state['global_step']!=expected:raise ValueError('Unexpected optimizer update budget')
    print('M30_SMOKE_COMPLETE' if args.smoke else 'M30_TRAINING_COMPLETE',args.variant,state['global_step'],
          'peak_gpu_gib',torch.cuda.max_memory_allocated()/1024**3,flush=True)


if __name__=='__main__':main()
