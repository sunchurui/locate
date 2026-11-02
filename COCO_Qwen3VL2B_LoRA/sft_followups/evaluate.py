"""Native/highres crossed inference using the existing Rex parser and current scorer."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from core import apply_resolution


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--adapter',required=True)
    p.add_argument('--output',required=True);p.add_argument('--profile',choices=['native','highres'],required=True)
    p.add_argument('--manifest');args=p.parse_args();root=Path(args.root).resolve()
    sys.path.insert(0,str(root/'sft_ablation'))
    spec=importlib.util.spec_from_file_location('sft_eval',root/'sft_ablation/evaluate.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    if torch.cuda.get_device_capability()[0]<8:raise ValueError('Use Ampere/newer BF16 inference')
    config=json.loads((root/'sft_followups/config.json').read_text())
    processor=AutoProcessor.from_pretrained(args.adapter,local_files_only=True)
    apply_resolution(processor,args.profile,config['highres'])
    assert [processor.tokenizer.convert_tokens_to_ids(f'<{i}>') for i in range(1000)]==list(range(150643,151643))
    base=Qwen3VLForConditionalGeneration.from_pretrained(root/'model',torch_dtype=torch.bfloat16,
        local_files_only=True,device_map='cuda')
    model=PeftModel.from_pretrained(base,args.adapter).eval()
    evaluator.run(model,processor,root,Path(args.output),'image_first',16,1024,args.manifest)


if __name__=='__main__':main()
