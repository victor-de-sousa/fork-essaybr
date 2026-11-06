"""Read-only evidence from training/development and existing artifacts; no test metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from .dataset import file_hash, load_records, normalize, verify_prepared, write_json
from .metrics import score_metrics


def qwk_from_counts(matrix):
    matrix = np.asarray(matrix, dtype=float)
    weights = (np.arange(len(matrix))[:, None] - np.arange(len(matrix))) ** 2
    expected = np.outer(matrix.sum(1), matrix.sum(0)) / matrix.sum()
    denominator = (weights * expected).sum()
    return float(1 - (weights * matrix).sum() / denominator) if denominator else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = Path('datasets/prepared')
    verify_prepared(data)
    train, dev = [load_records(data / f'{s}.jsonl') for s in ('training', 'development')]
    result = {'test_used': False, 'runs': {}, 'preserved_sha256': {}}
    # Hashing the held-out file establishes immutability; do not parse its targets.
    for root in ('artifacts/models', 'datasets/prepared', 'essay-br/splits'):
        for path in sorted(Path(root).rglob('*')):
            if path.is_file():
                result['preserved_sha256'][str(path)] = file_hash(path)
    result['train_dev_exact_overlap'] = len({normalize(r['text']) for r in train} & {normalize(r['text']) for r in dev})
    result['constant_120_validation'] = score_metrics([r['scores'] for r in dev], np.full((len(dev), 5), 120))
    for name in ('baseline', 'bertimbau-frozen'):
        path = Path('artifacts/models') / name
        meta = json.loads((path / 'metadata.json').read_text())
        log = json.loads((path / 'training_log.json').read_text())
        run = {'metadata': meta, 'epochs': []}
        for entry in log.get('epochs', []):
            diagnostics = {}
            for c, metric in entry['validation_metrics']['competencies'].items():
                cm = np.array(metric['confusion_matrix'])
                diagnostics[c] = {'truth': cm.sum(1).tolist(), 'predicted': cm.sum(0).tolist(),
                                  'qwk_independent': qwk_from_counts(cm), 'qwk_logged': metric['qwk']}
            run['epochs'].append({k: v for k, v in entry.items() if k != 'validation_metrics'} | {
                'macro': entry['validation_metrics']['macro'], 'distributions': diagnostics})
        result['runs'][name] = run
    with safe_open('artifacts/models/bertimbau-frozen/model.safetensors', framework='pt') as trained, safe_open('artifacts/models/bertimbau/model.safetensors', framework='pt') as base:
        encoder_keys = [k for k in trained.keys() if k.startswith('encoder.')]
        result['encoder_identical_to_base'] = all(torch.equal(trained.get_tensor(k), base.get_tensor(k.removeprefix('encoder.'))) for k in encoder_keys)
        first, increments = trained.get_tensor('first_threshold'), trained.get_tensor('threshold_increments')
        result['best_thresholds'] = torch.cat((first, first + torch.nn.functional.softplus(increments).cumsum(-1)), -1).tolist()
    result['resume_states'] = {}
    paths = [Path('artifacts/models/bertimbau-frozen/checkpoint-last/trainer.pt')]
    paths.extend(Path('artifacts/models/training-backups/overnight-frozen').glob('*/trainer_state.pt'))
    for path in paths:
        state = torch.load(path, weights_only=True, map_location='cpu')
        result['resume_states'][str(path)] = {k: state.get(k) for k in ('epoch', 'next_batch', 'global_step', 'stale')}
        result['resume_states'][str(path)]['indices_length'] = len(state['indices']) if state.get('indices') is not None else None
        result['resume_states'][str(path)]['scheduler_last_epoch'] = state['scheduler']['last_epoch']
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('runs', 'preserved_sha256', 'constant_120_validation')}))


if __name__ == '__main__':
    main()
