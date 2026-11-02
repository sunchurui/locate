"""CPU preflight using real prepared rows, tokenizer, images and the reused collator."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sft_ablation'))
from transformers import AutoProcessor
from torch.utils.data import SequentialSampler


def main():
    report=json.loads((ROOT/'sft_curriculum/prepared/report.json').read_text())
    rows=[json.loads(line) for line in (ROOT/'sft_curriculum/prepared/curriculum.jsonl').open()]
    control=json.loads((ROOT/'sft_curriculum/prepared/random_control_ids.json').read_text())
    assert {v['image_id'] for v in rows}==set(control) and len(rows)==len(control)==118287
    actual=[rows[i]['image_id'] for i in SequentialSampler(rows)]
    assert actual==json.loads((ROOT/'sft_curriculum/prepared/curriculum_ids.json').read_text())
    spec=importlib.util.spec_from_file_location('sft_helpers',ROOT/'sft_ablation/train.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    processor=AutoProcessor.from_pretrained(ROOT/'rex_id_tokenizer',local_files_only=True)
    processor.image_processor.max_pixels=1024*28*28
    processor.image_processor.min_pixels=16*28*28
    processor.image_processor.size.update(longest_edge=1024*28*28,shortest_edge=16*28*28)
    assert [processor.tokenizer.convert_tokens_to_ids(f'<{q}>') for q in range(1000)]==list(range(150643,151643))
    chosen=[rows[0],max(rows,key=lambda v:len(v['answer']))]
    collator=helper.Collator(processor,4096,'image_first');batch=collator(chosen)
    assert batch['input_ids'].shape==batch['labels'].shape
    assert (batch['labels']!=-100).sum()>0
    result={'real_image_ids':[v['image_id'] for v in chosen],
        'padded_shape':list(batch['input_ids'].shape),'visual_grid':batch['image_grid_thw'].tolist(),
        'coordinate_ids_verified':True,'sequential_order_verified':True,
        'same_unique_image_set_verified':True,'phase1_end_step':report['phase1_end_step']}
    (ROOT/'sft_curriculum/preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('M30_REAL_INPUT_PREFLIGHT_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
