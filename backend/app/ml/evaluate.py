"""Evaluate a frozen artifact. Held-out testing requires the explicit --final-test flag."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
from .dataset import file_hash,load_records,write_json
from .metrics import quantize,score_metrics
from .predict import Predictor,ModelUnavailableError


def evaluate(data_dir: Path,model_dir: Path,split: str,output: Path,final_test: bool=False):
    if split=='testing' and not final_test:
        raise ValueError('Testing is reserved: pass --final-test only after validation selection is frozen')
    if split not in ('development','testing'):
        raise ValueError('Evaluation split must be development or testing')
    data_dir,model_dir,output=Path(data_dir),Path(model_dir),Path(output)
    predictor=Predictor(model_dir);metadata=predictor.status()
    if not metadata['trained']: raise ModelUnavailableError(metadata['message'])
    # Stop evaluation of an artifact against a different training/validation dataset.
    for used_split,expected in metadata.get('data_sha256',{}).items():
        if file_hash(data_dir/f'{used_split}.jsonl')!=expected:
            raise ValueError(f'Dataset provenance mismatch for {used_split}')
    manifest_path=data_dir/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if file_hash(data_dir/f'{split}.jsonl')!=manifest['files'][split]['sha256']:
            raise ValueError('Evaluation data do not match the prepared manifest')
    records=load_records(data_dir/f'{split}.jsonl')
    predictions=quantize(predictor.predict_raw([r['text'] for r in records]))
    metrics=score_metrics([r['scores'] for r in records],predictions)
    report={'schema_version':1,'evaluated_at':datetime.now(timezone.utc).isoformat(),
        'split':split,'final_test':split=='testing','model_version':metadata['version'],
        'kind':metadata['kind'],'selection_split':metadata.get('selection_split'),
        'data_sha256':file_hash(data_dir/f'{split}.jsonl'),'metrics':metrics,
        'predictions':[{'id':r['id'],'reference':r['scores'],'prediction':p.tolist()} for r,p in zip(records,predictions)],
        'policy':'Test results must not be used to tune hyperparameters or choose a model.'}
    write_json(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('datasets/prepared'))
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--split',choices=['development','testing'],default='development')
    p.add_argument('--final-test',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=evaluate(args.data,args.model,args.split,args.output,args.final_test)
    print(json.dumps({'split':report['split'],'version':report['model_version'],'metrics':report['metrics']['macro']},indent=2))

if __name__=='__main__': main()
