"""Train a real TF-IDF/Ridge baseline; alpha selection sees development only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from .dataset import file_hash, load_records, write_json, verify_prepared
from .metrics import score_metrics


def selection_key(metrics):
    qwk = metrics['macro']['qwk']
    return (-qwk if qwk is not None else float('inf'), metrics['macro']['mae'])


def train_baseline(data_dir: Path, output: Path, alphas=(1.,10.,100.), seed: int = 42, max_features: int = 50000):
    data_dir, output = Path(data_dir), Path(output)
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError('Ridge alphas must be positive')
    verify_prepared(data_dir)
    training = load_records(data_dir / 'training.jsonl')
    development = load_records(data_dir / 'development.jsonl')
    if not training or not development:
        raise ValueError('Training and development sets must both be nonempty')
    x = [r['text'] for r in training]; y = np.array([r['scores'] for r in training])
    dev_x = [r['text'] for r in development]; dev_y = np.array([r['scores'] for r in development])
    candidates, best_model, best_metrics, selected_alpha = [], None, None, None
    started = time.monotonic()
    for alpha in alphas:
        model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=max_features, sublinear_tf=True)),
                          ('ridge', Ridge(alpha=alpha, solver='lsqr', random_state=seed))])
        model.fit(x,y)
        metrics = score_metrics(dev_y, model.predict(dev_x))
        candidates.append({'alpha': alpha, 'metrics': metrics})
        if best_metrics is None or selection_key(metrics) < selection_key(best_metrics):
            best_model, best_metrics, selected_alpha = model, metrics, alpha
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output/'baseline.joblib')
    provenance = {split: file_hash(data_dir / f'{split}.jsonl') for split in ('training','development')}
    version = 'tfidf-ridge-' + hashlib.sha256(json.dumps({'data':provenance, 'alpha':selected_alpha, 'seed':seed},sort_keys=True).encode()).hexdigest()[:12]
    metadata = {'schema_version':1, 'version':version, 'trained':True, 'kind':'tfidf_ridge',
        'created_at':datetime.now(timezone.utc).isoformat(), 'seed':seed,
        'selection_split':'development', 'selection_metric':'macro_qwk_then_macro_mae',
        'train_count':len(training), 'development_count':len(development), 'data_sha256':provenance,
        'hyperparameters':{'alpha':selected_alpha,'max_features':max_features,'ngram_range':[1,2]},
        'validation_metrics':best_metrics, 'sklearn_version':sklearn.__version__,
        'artifact_sha256':file_hash(output/'baseline.joblib'), 'test_evaluated':False}
    write_json(output/'metadata.json',metadata)
    report = {'selection_split':'development','selection_metric':metadata['selection_metric'],
        'selected_alpha':selected_alpha,'candidates':candidates,'elapsed_seconds':time.monotonic()-started,
        'version':version,'test_evaluated':False}
    write_json(output/'training_log.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('datasets/prepared'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/models/baseline'))
    parser.add_argument('--alphas',nargs='+',type=float,default=[1.,10.,100.])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-features',type=int,default=50000)
    args=parser.parse_args()
    report=train_baseline(args.data,args.output,args.alphas,args.seed,args.max_features)
    best=next(c for c in report['candidates'] if c['alpha']==report['selected_alpha'])
    print(json.dumps({'version':report['version'],'alpha':report['selected_alpha'],'development':best['metrics']['macro'],'test_evaluated':False},indent=2))

if __name__=='__main__': main()
