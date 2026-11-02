"""Small adaptations shared by the four single-factor Rex-ID SFT experiments."""
import copy
import hashlib
import json

import torch
import torch.nn.functional as F


LANGUAGE_TARGET=r'.*language_model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
PROJECTOR_TARGET=r'.*visual\.(merger|deepstack_merger_list\.\d+)\.linear_fc[12]'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def apply_resolution(processor,profile,settings=None):
    image=processor.image_processor
    if profile=='native':minimum,maximum=16*28*28,1024*28*28
    elif profile=='highres':
        settings=settings or {'min_visual_tokens':1024,'max_visual_tokens':1280}
        factor=image.patch_size*image.merge_size
        minimum=settings['min_visual_tokens']*factor**2
        maximum=settings['max_visual_tokens']*factor**2
    else:raise ValueError(profile)
    image.min_pixels=minimum;image.max_pixels=maximum
    image.size.update(shortest_edge=minimum,longest_edge=maximum)
    return {'profile':profile,'min_pixels':minimum,'max_pixels':maximum,
        'patch_size':image.patch_size,'merge_size':image.merge_size}


def image_mean_loss(outputs,labels,num_items_in_batch=None):
    """Nemotron sample-mean formula; divisor covers IMAGES in the whole GA window."""
    logits=outputs['logits'] if isinstance(outputs,dict) else outputs.logits
    sums=[]
    for i in range(labels.shape[0]):
        target=labels[i,1:];valid=target.ne(-100)
        if not valid.any():raise ValueError('Image has no supervised output token')
        # Select supervised rows before upcasting; never materialize full fp32 logits.
        sums.append(F.cross_entropy(logits[i,:-1][valid].float(),target[valid],reduction='mean'))
    divisor=labels.shape[0] if num_items_in_batch is None else num_items_in_batch
    return torch.stack(sums).sum()/divisor


class ImageCountMixin:
    def _get_num_items_in_batch(self,batch_samples,device):
        return torch.tensor(sum(batch['labels'].shape[0] for batch in batch_samples),device=device)


def common_hash(parameters):
    digest=hashlib.sha256()
    for name,value in parameters.items():
        digest.update(name.encode());digest.update(value.detach().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_projector_adapter(model,config,original_builder):
    """Preserve common LoRA initialization exactly while extending target modules.

    A direct larger regex consumes RNG in visual modules before language modules.
    Save the standard adapter's fresh trainables, unload WITHOUT merging, create
    the expanded adapter, then restore every common trainable tensor verbatim.
    """
    common=original_builder(model,copy.deepcopy(config))
    initial={name:p.detach().clone() for name,p in common.named_parameters() if p.requires_grad}
    before=common_hash(initial)
    base=common.unload()  # No optimization occurred; unload without merging any adapter.
    extended=copy.deepcopy(config)
    extended.target_modules=f'({LANGUAGE_TARGET})|({PROJECTOR_TARGET})'
    result=original_builder(base,extended)
    parameters=dict(result.named_parameters())
    with torch.no_grad():
        for name,value in initial.items():
            if name not in parameters:raise ValueError(f'Common adapter tensor disappeared: {name}')
            parameters[name].copy_(value)
    after=common_hash({name:parameters[name] for name in initial})
    if before!=after:raise ValueError('Shared initialization changed')
    extra=[name for name,p in result.named_parameters() if p.requires_grad and name not in initial]
    if len(extra)!=16 or any('.visual.' not in name or '.lora_' not in name for name in extra):
        raise ValueError(f'Expected A/B in8 merger linear layers, got: {extra}')
    if any(p.requires_grad for name,p in result.named_parameters() if '.visual.blocks.' in name):
        raise ValueError('Vision backbone must remain frozen')
    result._followup_initialization={'common_sha256':before,'common_restored_sha256':after,
        'extra_trainables':extra,'common_trainable_parameters':sum(v.numel() for v in initial.values()),
        'extra_trainable_parameters':sum(parameters[n].numel() for n in extra)}
    return result
