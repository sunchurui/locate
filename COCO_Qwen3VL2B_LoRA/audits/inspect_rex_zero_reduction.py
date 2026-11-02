import sys,json,math,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from official_metrics import author_module,load_predictions
root=Path(sys.argv[1]);data=json.loads((root/'data/coco2017/annotations/instances_val2017.json').read_text());pred=load_predictions(root/'evaluation/rex_id_val2017_a800_80/predictions.jsonl');fixed=json.loads((root/'rl_val_manifest.json').read_text())['image_ids'];author=author_module();report={}
for name,ids in [('full5000',sorted(pred)),('fixed256',fixed)]:
 wanted=set(ids);gt={'info':{},'images':sorted([i for i in data['images'] if i['id'] in wanted],key=lambda i:i['id']),'categories':data['categories'],'annotations':[a for a in data['annotations'] if a['image_id'] in wanted]};dt=[dict(p,image_id=i,score=.2) for i in sorted(ids) for p in pred[i]]
 with tempfile.TemporaryDirectory(dir=root/'runtime_tmp') as tmp:
  g=Path(tmp)/'gt.json';d=Path(tmp)/'pred.json';g.write_text(json.dumps(gt));d.write_text(json.dumps(dt));raw=author.fe.evaluate(str(g),str(d),0,0,'coco')
 report[name]={}
 for key in ['precision','recall','precision50','recall50','precision95','recall95']:
  values=raw[key];report[name][key]={'total_entries':len(values),'positive_entries':sum(v>0 for v in values),'zero_entries':sum(v==0 for v in values),'negative_entries':sum(v<0 for v in values),'nonfinite_entries':sum(not math.isfinite(v) for v in values),'author_safe_mean':author.safe_mean(values)}
print('RAW_REDUCTION',json.dumps(report),flush=True)
(root/'audits/official_subset_f1_20261102/raw_reduction.json').write_text(json.dumps(report,indent=2)+'\n')
