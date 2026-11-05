"""Metrics on the six ENEM score levels, with undefined QWK represented as null."""
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def quantize(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Scores must be finite')
    return (np.floor(np.clip(values, 0, 200) / 40 + .5) * 40).astype(int)


def _one(truth, prediction, maximum):
    labels = np.arange(0, maximum + 1, 40)
    error = prediction - truth
    # Kappa is undefined when the expected disagreement is zero.
    combined = np.unique(np.concatenate((truth, prediction)))
    qwk = float(cohen_kappa_score(truth, prediction, labels=labels, weights='quadratic')) if len(combined) > 1 else None
    if qwk is not None and not np.isfinite(qwk):
        qwk = None
    errors, counts = np.unique(error, return_counts=True)
    return {'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt((error.astype(float)**2).mean())),
            'qwk': qwk, 'exact_agreement': float(np.mean(error == 0)),
            'within_40': float(np.mean(np.abs(error) <= 40)),
            'labels': labels.tolist(), 'confusion_matrix': confusion_matrix(truth, prediction, labels=labels).tolist(),
            'error_distribution': {str(int(e)): int(n) for e, n in zip(errors, counts)}}


def score_metrics(y_true, y_pred):
    truth, prediction = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 5 or truth.shape != prediction.shape or not len(truth):
        raise ValueError('Expected equally shaped nonempty arrays (essays, 5 competencies)')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Scores must be finite')
    if not np.isin(truth, [0,40,80,120,160,200]).all():
        raise ValueError('Reference scores must use the ENEM scale 0..200, step 40')
    truth, prediction = truth.astype(int), quantize(prediction)
    competency = {f'c{i+1}': _one(truth[:,i], prediction[:,i], 200) for i in range(5)}
    kappas = [x['qwk'] for x in competency.values() if x['qwk'] is not None]
    return {'n': len(truth), 'competencies': competency, 'total': _one(truth.sum(1), prediction.sum(1), 1000),
            'macro': {'mae': float(np.mean([m['mae'] for m in competency.values()])),
                      'rmse': float(np.mean([m['rmse'] for m in competency.values()])),
                      'qwk': float(np.mean(kappas)) if kappas else None},
            'quantization': 'clip 0..200; nearest 40; halfway rounds up'}
