"""Launch ms-swift SFT with a fallback for PyTorch builds before FSDP2."""

import runpy

import torch.distributed.fsdp as fsdp

# ms-swift imports this optional FSDP2 class even for ordinary single-GPU runs.
# On PyTorch versions without FSDP2, the legacy FSDP class is a safe fallback.
if not hasattr(fsdp, "FSDPModule"):
    fsdp.FSDPModule = fsdp.FullyShardedDataParallel

runpy.run_module("swift.cli.sft", run_name="__main__")
