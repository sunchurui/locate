import contextlib,io,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pycocotools.coco import COCO
from metrics import evaluate_records
root=Path(__file__).resolve().parent.parent
with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(root/'data/coco2017/annotations/instances_val2017.json'))
report={'time':time.strftime('%Y-%m-%d %H:%M:%S %Z'),'note':'INTERIM unequal image subsets; paired baseline within each method only. No full-val or final claim.','methods':{}}
for method in ('m06','m44','m45','m46'):
 records=[json.loads(p.read_text()) for p in sorted((root/'inference_ablation/results'/method).glob('shard*/records/*.json'))]
 if not records:continue
 metrics=evaluate_records(gt,records)
 item={'images':len(records),'metrics':{name:{'meanIoU':values['meanIoU'],'IoU50':values['IoU50'],'IoU95':values['IoU95'],'delta_F1_meanIoU_points':values.get('delta_F1_meanIoU_points',0),'detections':values['detections']} for name,values in metrics.items()},'baseline_cost':{k:sum(r['baseline_cost'][k] for r in records) for k in records[0]['baseline_cost']},'extra_cost':{k:sum(r['extra_cost'][k] for r in records) for k in records[0]['extra_cost']}}
 if method=='m44':
  details=[d for r in records for d in r['details']];item['diagnostics']={'initial_boxes':len(details),'accepted':sum(d['accepted'] for d in details)}
 if method=='m46':
  ds=[r['details'] for r in records];item['diagnostics']={'strict_integer':sum(d['strict_integer_response'] for d in ds),'conflicts':sum(d['conflict'] for d in ds),'nonempty_rechecks':sum(bool(d['recheck_count']) for d in ds),'changed_counts':sum(d['initial_count']!=d['final_count'] for d in ds)}
 report['methods'][method]=item
 print(method,json.dumps(item),flush=True)
(root/'inference_ablation/results/interim_rex_official_current.json').write_text(json.dumps(report,indent=2)+'\n')
