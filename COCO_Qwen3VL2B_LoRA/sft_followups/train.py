"""Four separate, equal-budget SFT ablations reusing the frozen Rex-ID trainer."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import torch
from torch.utils.data import SequentialSampler
from transformers import set_seed

from core import apply_resolution,build_projector_adapter,common_hash,image_mean_loss,ImageCountMixin,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--variant',choices=['image_mean','projector_lora','highres','class_rfs'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();root=Path(args.root).resolve();folder=root/'sft_followups'
    config=json.loads((folder/'config.json').read_text());report=json.loads((folder/'prepared/report.json').read_text())
    if config!=report['config']:raise ValueError('Frozen config differs')
    for name,digest in report['shared_sources'].items():
        if sha(root/name)!=digest:raise ValueError(f'Shared source changed: {name}')
    if sha(folder/'sources.lock.json')!=report['source_lock_sha256']:raise ValueError('Source lock changed')
    data=folder/'prepared'/('class_rfs.jsonl' if args.variant=='class_rfs' else 'common_order.jsonl')
    if sha(data)!=report['files'][data.name]:raise ValueError('Prepared rows changed')
    sys.path.insert(0,str(root/'sft_ablation'))
    import baseline_train as baseline
    spec=importlib.util.spec_from_file_location('sft_helpers',root/'sft_ablation/train.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    from evaluate import run as generation_evaluate
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<80_000_000_000:
        raise ValueError('Training/smoke require >=80GB GPU')
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Preserve existing outputs; choose a fresh directory')
    output.mkdir(parents=True,exist_ok=True)
    shared=json.loads((root/'sft_ablation/config.json').read_text())
    profile='highres' if args.variant=='highres' else 'native'
    metadata={'method':'sft_followup_'+args.variant,'variant':args.variant,'input_order':'image_first',
        'config':shared,'followup_config':config,'report_sha256':sha(folder/'prepared/report.json'),
        'train_sha256':sha(data),'smoke':args.smoke,'training_resolution':profile,
        'initialization':'fresh pretrained Qwen3-VL-2B + seeded language LoRA + original Rex token rows',
        'control':config['control'],'evaluation_protocol':config['evaluation_protocol'],
        'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory}
    snapshot=output/'source';snapshot.mkdir()
    for source in list(folder.glob('*.py'))+list(folder.glob('*.json'))+list(folder.glob('*.sbatch'))+[folder/'README.md']:
        if source.is_file():shutil.copy2(source,snapshot/source.name)

    class Collator(helpers.Collator):
        def __init__(self,processor,max_length):
            metadata['resolution_settings']=apply_resolution(processor,profile,config['highres'])
            super().__init__(processor,max_length,'image_first')

    class Protocol(helpers.Protocol):
        def on_pre_optimizer_step(self,training_args,state,control,model=None,**kwargs):
            super().on_pre_optimizer_step(training_args,state,control,model=model,**kwargs)
            if args.smoke and args.variant=='projector_lora':
                grads=[v.grad for n,v in model.named_parameters() if '.visual.' in n and '.lora_' in n and v.grad is not None]
                if not grads or not any(torch.count_nonzero(g).item() for g in grads):raise ValueError('Projector LoRA has no training signal')
            return control

        def on_save(self,training_args,state,control,model=None,**kwargs):
            checkpoint=output/f'checkpoint-{state.global_step}'
            (checkpoint/'sft_ablation.json').write_text(json.dumps(metadata,indent=2)+'\n')
            self.processor.save_pretrained(checkpoint)
            if not args.smoke and state.global_step in config['train']['generation_steps']:
                # Primary comparison always uses native inference resolution.
                profiles=['native','highres'] if args.variant=='highres' else ['native']
                try:
                    for inference_profile in profiles:
                        apply_resolution(self.processor,inference_profile,config['highres'])
                        metrics=generation_evaluate(model,self.processor,root,
                            output/'progress_eval'/f'step-{state.global_step}_{inference_profile}_val256',
                            'image_first',16,1024,root/'rl_val_manifest.json')
                        with (output/'progress_micro_metrics.jsonl').open('a') as f:
                            f.write(json.dumps({'step':state.global_step,'inference_resolution':inference_profile,**metrics})+'\n')
                finally:apply_resolution(self.processor,profile,config['highres'])
            return control

    callback=Protocol(root,metadata,None,args.smoke)
    if args.variant=='projector_lora':
        original_builder=baseline.get_peft_model
        baseline.get_peft_model=lambda model,lora_config:build_projector_adapter(model,lora_config,original_builder)
    original=baseline.Trainer

    class VariantTrainer(original):
        def _get_train_sampler(self,train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def _get_num_items_in_batch(self,batch_samples,device):
            if args.variant=='image_mean':return ImageCountMixin._get_num_items_in_batch(self,batch_samples,device)
            return super()._get_num_items_in_batch(batch_samples,device)

        def __init__(self,*positional,**kwargs):
            training=kwargs['args'];training.max_steps=2 if args.smoke else config['train']['updates']
            training.save_total_limit=None;training.dataloader_num_workers=0 if args.smoke else 2
            if args.smoke:
                training.eval_strategy='no';training.save_strategy='no';training.logging_steps=1
            if args.variant=='image_mean':kwargs['compute_loss_func']=image_mean_loss
            callback.processor=kwargs['processing_class']
            kwargs['callbacks']=list(kwargs.pop('callbacks',[]))+[callback]
            model=kwargs['model']
            initial={name:v for name,v in model.named_parameters() if v.requires_grad}
            metadata['initial_trainable_parameters_sha256']=common_hash(initial)
            metadata['trainable_parameters']=sum(v.numel() for v in initial.values())
            if args.variant=='projector_lora':metadata['projector_initialization']=model._followup_initialization
            (output/'sft_ablation.json').write_text(json.dumps(metadata,indent=2)+'\n')
            print('SFT_FOLLOWUP_PROTOCOL',json.dumps(metadata),flush=True)
            super().__init__(*positional,**kwargs)

    baseline.Trainer=VariantTrainer;baseline.CocoRexCollator=Collator
    settings=config['train'];set_seed(42)
    sys.argv=[str(root/'sft_ablation/baseline_train.py'),'--model',str(root/'model'),
        '--tokenizer-dir',str(root/'rex_id_tokenizer'),'--train-data',str(data),
        '--val-data',str(root/'data/coco2017/annotations/val_rex_token.jsonl'),
        '--output-dir',str(output),'--train-batch-size',str(settings['batch']),
        '--eval-batch-size','4','--grad-accum',str(settings['gradient_accumulation']),
        '--epochs','1','--learning-rate',str(settings['learning_rate']),
        '--eval-steps','1000','--save-steps','1000','--max-length','4096']
    baseline.main()
    state=json.loads((output/'trainer_state.json').read_text())
    assert state['global_step']==(2 if args.smoke else settings['updates'])
    print('SFT_FOLLOWUP_SMOKE_COMPLETE' if args.smoke else 'SFT_FOLLOWUP_TRAIN_COMPLETE',
        args.variant,state['global_step'],'peak_gpu_gib',torch.cuda.max_memory_allocated()/1024**3,flush=True)


if __name__=='__main__':main()
