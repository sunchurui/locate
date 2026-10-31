"""Rex token parser and GroundingPI-inspired dual rewards for ms-swift."""
import os
import re
import sys
from pathlib import Path

ROOT=Path(os.environ['EXPERIMENT_ROOT'])
sys.path.insert(0,str(ROOT/'references/Rex-Omni/finetuning'))
sys.path.insert(0,str(ROOT/'rex_groundingpi_adaptation'))
import groundingpi_reward as core
from verl.configs.reward_func import BoxIoURewardFunction
from swift.rewards import ORM, orms

REX=BoxIoURewardFunction()


def parse(text):
    text=text.strip()
    for tail in ('<|im_end|>','<|endoftext|>'):
        if text.endswith(tail):
            text=text[:-len(tail)].strip()
    if text=='None':
        return [],True
    pattern=r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>(.*?)<\|box_end\|>'
    matches=list(re.finditer(pattern,text,re.S))
    if not matches:
        return [],False
    pos=0
    for match in matches:
        if text[pos:match.start()].strip(' ,\n\t'):
            return [],False
        pos=match.end()
        label,payload=match.group(1).strip().lower(),match.group(2).strip()
        if not label:
            return [],False
        if payload=='None':
            continue
        for boxtext in payload.split(','):
            if not re.fullmatch(r'(?:<\d{1,3}>){4}',boxtext.strip()):
                return [],False
            box=[int(v)*1000/999 for v in re.findall(r'<(\d+)>',boxtext)]
            if box[2]<=box[0] or box[3]<=box[1]:
                return [],False
    if text[pos:].strip():
        return [],False
    # Reuse the official Rex extractor after strict format validation. Its
    # q/1000 decoding is converted to the q/999 convention of our SFT/eval.
    extracted=REX.parse_detection_output(text,1000,1000)
    if extracted is None:
        return [],False
    return [(label.lower(),[v*1000/999 for v in box])
            for label,boxes in extracted['objects'].items() for box in boxes],True


def strict(pred,truth,valid):
    if not valid:return 0.0
    if not pred and not truth:return 1.0
    p,g=len(pred),len(truth)
    # User's proposed strict count term: exact agreement, not min/max.
    count=float(p==g)
    matches=[core._matching(pred,truth,t) for t in (0.5,0.75,0.95)]
    f1=sum(2*len(m)/(p+g) if p+g else 1.0 for m in matches)/3
    overlap=sum(matches[0])/len(matches[0]) if matches[0] else 0.0
    last={};ordered=True
    for label,box in pred:
        if box[0]<last.get(label,-1):ordered=False
        last[label]=box[0]
    # Count each later duplicate once; count penalties are not divided by p.
    # The .9 duplicate / .8 area / .5 support thresholds remain unconfirmed.
    dup=sum(any(label==old_label and core._iou(box,old_box)>=0.9 for old_label,old_box in pred[:i])
            for i,(label,box) in enumerate(pred))
    large=sum((b[2]-b[0])*(b[3]-b[1])>800000 and
              not any(label==gt_label and core._iou(b,gt_box)>=0.5 for gt_label,gt_box in truth)
              for label,b in pred)
    return max(0.,min(1.,.1+.1*count+.5*f1+.25*overlap+.05*ordered-.07*large-.03*dup))


class SetReward(ORM):
    def __call__(self,completions,ground_truth,**kwargs):
        if len(completions)!=len(ground_truth):raise ValueError('reward batch mismatch')
        result=[]
        for text,gt in zip(completions,ground_truth):
            pred,valid=parse(text)
            result.append(core._set_reward(pred,core._parse_truth(gt)) if valid else 0.)
        return result


class StrictReward(ORM):
    def __call__(self,completions,ground_truth,**kwargs):
        if len(completions)!=len(ground_truth):raise ValueError('reward batch mismatch')
        result=[]
        for text,gt in zip(completions,ground_truth):
            pred,valid=parse(text)
            result.append(strict(pred,core._parse_truth(gt),valid))
        return result


orms['rex_id_set']=SetReward
orms['rex_id_strict']=StrictReward
