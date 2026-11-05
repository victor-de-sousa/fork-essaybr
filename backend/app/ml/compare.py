"""Compare completed baseline, ordinal and regression runs using validation only."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from .baseline import selection_key
from .dataset import file_hash,write_json
from .predict import Predictor


def compare(model_dirs,output):
    candidates=[];provenance=None
    for directory in model_dirs:
        directory=Path(directory)
        metadata=Predictor(directory).status()
        if not metadata.get('trained') or metadata.get('selection_split')!='development':
            raise ValueError(f'{directory} is not a trained validation-selected artifact')
        if 'validation_metrics' not in metadata:
            raise ValueError(f'{directory} lacks validation metrics')
        current=metadata.get('data_sha256')
        if not current or (provenance is not None and current!=provenance):
            raise ValueError('Candidates must share training and development provenance')
        provenance=current
        artifact=directory/('baseline.joblib' if metadata['kind']=='tfidf_ridge' else 'model.safetensors')
        candidates.append({'model_dir':str(directory.resolve()),'version':metadata['version'],
                           'kind':metadata['kind'],'metrics':metadata['validation_metrics'],
                           'artifact_sha256':file_hash(artifact)})
    if not candidates: raise ValueError('Provide at least one trained model')
    winner=min(candidates,key=lambda item:selection_key(item['metrics']))
    report={'selected_at':datetime.now(timezone.utc).isoformat(),'selection_split':'development',
            'selection_metric':'macro_qwk_then_macro_mae','data_sha256':provenance,
            'selected':winner,'candidates':candidates,'test_used':False}
    write_json(Path(output),report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('artifacts/reports/model_selection.json'))
    args=p.parse_args();result=compare(args.models,args.output)
    print(json.dumps({'selected':result['selected']['model_dir'],'version':result['selected']['version'],'test_used':False},indent=2))

if __name__=='__main__': main()
