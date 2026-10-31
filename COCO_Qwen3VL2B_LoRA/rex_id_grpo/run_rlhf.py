"""Use the installed ms-swift GRPO trainer with the site's Torch compatibility."""
import runpy
import torch.distributed.fsdp as fsdp
if not hasattr(fsdp,'FSDPModule'):
    fsdp.FSDPModule=fsdp.FullyShardedDataParallel
runpy.run_module('swift.cli.rlhf',run_name='__main__')
