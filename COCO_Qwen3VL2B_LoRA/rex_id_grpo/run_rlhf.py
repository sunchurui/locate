"""Configure upstream GRPO for the paper batch; preserve its optimization loop."""
import inspect
import json
import runpy
import textwrap
from collections import Counter
from pathlib import Path

import torch.distributed.fsdp as fsdp
if not hasattr(fsdp,'FSDPModule'):
    fsdp.FSDPModule=fsdp.FullyShardedDataParallel

from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin
from swift.pipelines.train.rlhf import SwiftRLHF

SETTINGS=json.loads(Path(__file__).with_name('alignment.json').read_text())

# Upstream 4.6.0 caps the sampled reference KL at 10. The paper equation has
# no such cap. Replace only those two expressions, leaving its loss reduction,
# clipping, gradient accumulation and optimizer implementation unchanged.
method=GRPOTrainer._compute_loss_and_metrics
source=textwrap.dedent(inspect.getsource(method))
old=('safe_ratio = torch.clamp(ref_per_token_logps - per_token_logps, min=-20, max=20)\n'
     '        per_token_kl = torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-10, max=10)')
new=('safe_ratio = ref_per_token_logps - per_token_logps\n'
     '        per_token_kl = torch.exp(safe_ratio) - safe_ratio - 1')
if source.count(old)!=1:
    raise RuntimeError('Upstream KL implementation changed; review paper alignment before training')
scope={}
exec(compile(source.replace(old,new),'<GroundingPI equation 1 KL adaptation>','exec'),method.__globals__,scope)
GRPOTrainer._compute_loss_and_metrics=scope['_compute_loss_and_metrics']

original_model_prepare=SwiftRLHF._prepare_model_tokenizer


def prepare_model(self):
    args=self.args
    for name,expected in SETTINGS['published_optimization'].items():
        if name in ('distinct_prompts_per_update','responses_per_update','updates_per_rollout'):
            continue
        actual=getattr(args,name,None)
        if str(actual).removeprefix('torch.')!=str(expected) and actual!=expected:
            raise ValueError(f'GroundingPI alignment: {name}={actual!r}, expected {expected!r}')
    execution=SETTINGS['single_gpu_execution']
    # Swift's outer RLHFArguments is not the initialized HF TrainingArguments;
    # distributed_state lives on its constructed training_args instance.
    if args.training_args.world_size!=execution['world_size']:
        raise ValueError('Single-GPU batch alignment failed: world_size')
    for name in ('per_device_train_batch_size','gradient_accumulation_steps'):
        if getattr(args,name)!=execution[name]:
            raise ValueError(f'Single-GPU batch alignment failed: {name}')
    if args.num_iterations!=1 or args.use_vllm:
        raise ValueError('Expected one update per rollout and the configured HF chunked-generation backend')
    if not args.adapters or args.adapters!=args.ref_adapters:
        raise ValueError('Policy and frozen reference must load the same SFT adapter')
    if args.dataset_shuffle:
        raise ValueError('Use the pre-shuffled complete-batch epoch file with dataset_shuffle=false')
    if not SETTINGS['unpublished_details_confirmed']:
        raise RuntimeError('Unpublished GroundingPI details await user confirmation')
    original_model_prepare(self)


SwiftRLHF._prepare_model_tokenizer=prepare_model
original_prepare=RolloutTrainerMixin.prepare_rollout


def prepare_rollout(self):
    original_prepare(self)
    execution=SETTINGS['single_gpu_execution']
    self.engine.max_batch_size=execution['generation_chunk_size']
    print('GroundingPI batch: 32 prompts x 8 responses = 256; one update after 256 microbatches; '
          f'generation memory chunk={self.engine.max_batch_size}; full-batch GDPO; uncapped equation-1 KL',flush=True)


RolloutTrainerMixin.prepare_rollout=prepare_rollout
original_advantages=GRPOTrainer._compute_advantages


def compute_advantages(self,samples,rewards_per_func,batch_encoded_inputs):
    if self.model.training:
        # Swift hashes message text for prompt_id, which is identical for all
        # COCO images; use the preserved image_id column to verify grouping.
        image_ids=[sample.extra['image_id'] for sample in samples]
        groups=Counter(image_ids)
        if rewards_per_func.shape!=(256,2) or len(groups)!=32 or set(groups.values())!={8}:
            raise ValueError(f'Expected 32 complete groups of 8 and two rewards; '
                             f'got shape={rewards_per_func.shape}, group_sizes={dict(groups)}')
        if any(len(set(image_ids[i:i+8]))!=1 for i in range(0,256,8)):
            raise ValueError('Responses are not grouped contiguously in groups of eight by image')
        if not getattr(self,'_groundingpi_batch_logged',False):
            print('Actual rollout verified: 32 distinct prompts, 8 responses each, rewards shape [256,2]',flush=True)
            self._groundingpi_batch_logged=True
    return original_advantages(self,samples,rewards_per_func,batch_encoded_inputs)


GRPOTrainer._compute_advantages=compute_advantages
runpy.run_module('swift.cli.rlhf',run_name='__main__')
