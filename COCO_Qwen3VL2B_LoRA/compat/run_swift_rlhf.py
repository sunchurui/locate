"""Launch ms-swift RLHF on the cluster's pre-FSDP2 PyTorch build."""
import runpy
import torch.distributed.fsdp as fsdp

if not hasattr(fsdp, 'FSDPModule'):
    fsdp.FSDPModule = fsdp.FullyShardedDataParallel
runpy.run_module('swift.cli.rlhf', run_name='__main__')
