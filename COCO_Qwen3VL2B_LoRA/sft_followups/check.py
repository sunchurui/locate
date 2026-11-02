"""CPU integration checks: real Trainer accumulation, real images and Qwen projector PEFT."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch.utils.data import SequentialSampler
from transformers import AutoProcessor,Trainer,TrainingArguments,Qwen3VLConfig,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model

from core import apply_resolution,build_projector_adapter,image_mean_loss,ImageCountMixin,LANGUAGE_TARGET

ROOT=Path(__file__).resolve().parents[1]


def accumulation_check():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(32,8);self.head=torch.nn.Linear(8,32)
        def forward(self,input_ids,attention_mask=None,**kwargs):
            return {'logits':self.head(self.embedding(input_ids))}
    class SampleTrainer(ImageCountMixin,Trainer):
        def _get_train_sampler(self,train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
    rows=[]
    for length in [4,7,5,8]:
        ids=torch.arange(1,length+1);labels=ids.clone();labels[:2]=-100
        rows.append({'input_ids':ids,'labels':labels})
    def collate(examples):
        return {'input_ids':torch.nn.utils.rnn.pad_sequence([v['input_ids'] for v in examples],batch_first=True),
                'labels':torch.nn.utils.rnn.pad_sequence([v['labels'] for v in examples],batch_first=True,padding_value=-100)}
    states=[]
    for micro,acc in [(3,1),(1,3)]:
        torch.manual_seed(91);model=Model();optimizer=torch.optim.SGD(model.parameters(),lr=.05)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime_tmp') as temporary:
            args=TrainingArguments(output_dir=temporary,use_cpu=True,per_device_train_batch_size=micro,
                gradient_accumulation_steps=acc,max_steps=2,max_grad_norm=0,report_to=[],
                save_strategy='no',logging_strategy='no',disable_tqdm=True,remove_unused_columns=False)
            trainer=SampleTrainer(model=model,args=args,train_dataset=rows,data_collator=collate,
                compute_loss_func=image_mean_loss,optimizers=(optimizer,scheduler))
            trainer.train();states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    maximum=max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0])
    if maximum>1e-6:raise ValueError(f'GA changes image-balanced update: {maximum}')
    return maximum


def projector_check():
    config=Qwen3VLConfig(text_config={'vocab_size':64,'hidden_size':16,'intermediate_size':32,
        'num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':8,
        'max_position_embeddings':128,'tie_word_embeddings':True,'rope_scaling':{'rope_type':'default','mrope_section':[2,1,1],'mrope_interleaved':True}},
        vision_config={'depth':24,'hidden_size':16,'intermediate_size':32,'num_heads':2,'out_hidden_size':16,
            'patch_size':16,'temporal_patch_size':2,'spatial_merge_size':2,'num_position_embeddings':16,
            'deepstack_visual_indexes':[5,11,17]},image_token_id=63,vision_start_token_id=61,vision_end_token_id=62,
        tie_word_embeddings=True)
    model=Qwen3VLForConditionalGeneration(config)
    pretrained=model.get_input_embeddings().weight.detach().clone()
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',
        target_modules=LANGUAGE_TARGET,trainable_token_indices=list(range(8,16)),ensure_weight_tying=True)
    result=build_projector_adapter(model,lora,get_peft_model)
    assert torch.equal(result.get_input_embeddings().weight,pretrained)
    assert torch.equal(result.get_input_embeddings().weight,result.get_output_embeddings().weight)
    visual=result.get_base_model().model.visual
    loss=visual.merger(torch.randn(12,16)).sum();loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for n,p in visual.merger.named_parameters() if '.lora_' in n)
    assert all(p.grad is None for n,p in visual.blocks.named_parameters())
    with tempfile.TemporaryDirectory(dir=ROOT/'runtime_tmp') as path:
        result.save_pretrained(path)
        saved=json.loads((Path(path)/'adapter_config.json').read_text())
        assert 'deepstack_merger_list' in saved['target_modules']
    return result._followup_initialization


def image_check():
    sys.path.insert(0,str(ROOT/'sft_ablation'))
    spec=importlib.util.spec_from_file_location('helpers',ROOT/'sft_ablation/train.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    rows=[json.loads(line) for line in (ROOT/'sft_followups/prepared/common_order.jsonl').open()]
    chosen=[rows[0],max(rows,key=lambda row:len(row['answer']))]
    processor=AutoProcessor.from_pretrained(ROOT/'rex_id_tokenizer',local_files_only=True)
    assert [processor.tokenizer.convert_tokens_to_ids(f'<{q}>') for q in range(1000)]==list(range(150643,151643))
    collator=helper.Collator(processor,4096,'image_first')
    apply_resolution(processor,'native');native=collator(chosen)
    apply_resolution(processor,'highres');higher=collator(chosen)
    for a,b in zip(native['labels'],higher['labels']):assert torch.equal(a[a!=-100],b[b!=-100])
    native_tokens=(native['image_grid_thw'].prod(dim=1)//4).tolist()
    high_tokens=(higher['image_grid_thw'].prod(dim=1)//4).tolist()
    assert all(h>n for n,h in zip(native_tokens,high_tokens)),'Highres must change actual grids'
    report={'image_ids':[v['image_id'] for v in chosen],'native_visual_tokens':native_tokens,
        'highres_visual_tokens':high_tokens,'native_input_shape':list(native['input_ids'].shape),
        'highres_input_shape':list(higher['input_ids'].shape),'supervised_target_tokens_unchanged':True}
    return report


def main():
    (ROOT/'runtime_tmp').mkdir(exist_ok=True)
    result={'image_mean_accumulation_max_parameter_difference':accumulation_check(),
        'projector':projector_check(),'actual_images':image_check()}
    (ROOT/'sft_followups/preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FOUR_SFT_CPU_PREFLIGHT_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
