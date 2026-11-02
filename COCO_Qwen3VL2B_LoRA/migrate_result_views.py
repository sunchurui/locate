"""Refresh current result views after micro replay; archive superseded values."""
import hashlib
import json
import os
from pathlib import Path

from micro_metrics import PROTOCOL

ROOT=Path(__file__).resolve().parent


def atomic(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(text);temp.replace(path)


def archive(path):
    if not path.exists():return
    folder=path.parent/'legacy_metrics';folder.mkdir(exist_ok=True)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    destination=folder/f'{path.stem}_{sha}{path.suffix}'
    if not destination.exists():destination.write_bytes(path.read_bytes())


def replace_json(path,value):
    archive(path);atomic(path,json.dumps(value,indent=2)+'\n')


def main():
    source=ROOT/'evaluation/micro_summary.json'
    summary=json.loads(source.read_text())
    assert summary['protocol']==PROTOCOL
    for row in summary['results']:
        p=ROOT/row['metrics'];metrics=json.loads(p.read_text())
        for name in ['coco_metrics.json','pilot_metrics.json']:
            old=p.parent/name
            if old.exists():replace_json(old,metrics)
        old=p.parent/'rex_official_metrics.json'
        if old.exists():
            value=json.loads(old.read_text())
            value['reporting_status']='historical_superseded'
            value['superseded_by']='micro_metrics.json'
            atomic(old,json.dumps(value,indent=2)+'\n')
    # Preserve previous named protocols as clearly-marked historical references.
    for p in [ROOT/'evaluation/rex_official_summary.json',ROOT/'evaluation/unranked_summary.json']:
        if p.exists():
            value=json.loads(p.read_text());value['reporting_status']='historical_superseded'
            value['superseded_by']='micro_summary.json';atomic(p,json.dumps(value,indent=2)+'\n')
    for p in [ROOT/'evaluation/rex_official_summary.md',ROOT/'evaluation/unranked_summary.md']:
        if p.exists():
            banner='> HISTORICAL / SUPERSEDED. Current results: [micro_summary.md](micro_summary.md). Do not compare these old scores with current results.\n\n'
            text=p.read_text()
            if not text.startswith(banner):atomic(p,banner+text)
    inf=ROOT/'inference_ablation/results'
    for ext in ['json','md']:
        current=inf/f'micro_summary.{ext}'
        if current.exists():
            for name in ['summary','set_summary']:
                target=inf/f'{name}.{ext}';archive(target);atomic(target,current.read_text())
    old=inf/'rex_official_summary.json'
    if old.exists():
        value=json.loads(old.read_text());value['reporting_status']='historical_superseded'
        value['superseded_by']='micro_summary.json';atomic(old,json.dumps(value,indent=2)+'\n')
    old=inf/'rex_official_summary.md'
    if old.exists() and not old.read_text().startswith('> HISTORICAL'):
        atomic(old,'> HISTORICAL / SUPERSEDED. Current results: [micro_summary.md](micro_summary.md).\n\n'+old.read_text())
    for method in ['m06','m44','m45','m46']:
        for old_dir in ['rex_official','unranked']:
            for p in (inf/method/old_dir).glob('**/*.json'):
                if 'legacy_metrics' in p.parts:continue
                value=json.loads(p.read_text())
                value['reporting_status']='historical_superseded'
                value['superseded_by']=str((inf/method/'micro/summary.json').relative_to(ROOT))
                atomic(p,json.dumps(value,indent=2)+'\n')
    # Recompute shard-only files from their own saved records, never borrow full-set scores.
    from inference_ablation.metrics import evaluate_records
    dataset=json.loads((ROOT/'data/coco2017/annotations/instances_val2017.json').read_text())
    shards=0
    for method in ['m06','m44','m45','m46']:
        for shard in sorted((inf/method).glob('shard*')):
            records=[json.loads(p.read_text()) for p in sorted((shard/'records').glob('*.json'))]
            if not records:continue
            values=evaluate_records(dataset,records)
            result={'protocol':PROTOCOL,'images':len(records),'image_ids':sorted(v['image_id'] for v in records),
                    'scope':'this shard only; see global micro_summary for complete results','metrics':values}
            for name in ['set_metrics.json','coco_metrics.json','pilot_metrics.json']:
                if (shard/name).exists():replace_json(shard/name,result)
            atomic(shard/'micro_metrics.json',json.dumps(result,indent=2)+'\n');shards+=1
    # Replace recorded training curves only when matching saved prediction folders exist.
    curves=0
    for folder in (ROOT/'sft_ablation_outputs').glob('*'):
        items=[]
        for p in (folder/'progress_eval').glob('step-*/micro_metrics.json'):
            items.append({'step':int(p.parent.name.split('-')[-1]),**json.loads(p.read_text())})
        if items:
            target=folder/'progress_set_metrics.jsonl';archive(target)
            atomic(target,''.join(json.dumps(v)+'\n' for v in sorted(items,key=lambda v:v['step'])));curves+=1
    # All canonical report views use the same registry; old Markdown copies are flagged.
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    banner='> Current metric: rex_matching_global_endpoint_micro_v1. Scores in percent. Saved-prediction replay; partial generation is explicitly marked.\n\n'
    atomic(reports/'micro_current.md',banner+source.with_suffix('.md').read_text())
    atomic(reports/'micro_current.json',source.read_text())
    for p in reports.glob('*.md'):
        if p.name=='micro_current.md':continue
        text=p.read_text()
        notice='> HISTORICAL METRIC SNAPSHOT. Superseded by [micro_current.md](micro_current.md); do not use these scores for current comparisons.\n\n'
        if not text.startswith(notice):atomic(p,notice+text)
    manifest={'protocol':PROTOCOL,'saved_result_groups':len(summary['results']),
        'shards_recomputed':shards,'training_curves_recomputed':curves,
        'all_previous_registry_sources_covered':True,
        'jobs':{'build':207836,'verification':207839,'initial_rescore':207844},
        'current_summary':'evaluation/micro_summary.json'}
    old=ROOT/'evaluation/rex_official_summary.json'
    if old.exists():
        old_sources={r['source'] for r in json.loads(old.read_text())['results']}
        new_sources={r['source'] for r in summary['results']}
        manifest['all_previous_registry_sources_covered']=old_sources.issubset(new_sources)
        manifest['missing_previous_sources']=sorted(old_sources-new_sources)
        assert not manifest['missing_previous_sources']
    atomic(ROOT/'audits/micro_migration_manifest.json',json.dumps(manifest,indent=2)+'\n')
    print('MICRO_RESULT_VIEWS_MIGRATED',json.dumps(manifest),flush=True)


if __name__=='__main__':main()
