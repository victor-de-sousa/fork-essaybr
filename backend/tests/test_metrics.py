import numpy as np
import pytest
from app.ml.metrics import score_metrics, quantize


def test_known_error_and_perfect_metrics():
    true=np.array([[0,40,80,120,160],[40,80,120,160,200]])
    pred=true.copy();pred[0,0]=40
    result=score_metrics(true,pred)
    assert result['competencies']['c1']['mae']==20
    assert result['competencies']['c1']['rmse']==pytest.approx(800**.5)
    assert result['competencies']['c1']['exact_agreement']==.5
    assert result['competencies']['c1']['within_40']==1
    assert result['competencies']['c2']['qwk']==1
    assert result['total']['mae']==20
    assert len(result['competencies']['c1']['confusion_matrix'])==6


def test_quantize_clips_and_rounds_half_up():
    assert quantize([-10,19,20,199,230]).tolist()==[0,0,40,200,200]


def test_metrics_reject_invalid_shape_and_empty():
    for value in ([],[[1,2]],[[0,40,80,120,float('nan')]]):
        with pytest.raises(ValueError): score_metrics(value,value)


@pytest.mark.parametrize('truth,prediction,expected', [
    ([0,40,80,120,160,200], [0,40,80,120,160,200], 1.),
    ([0,40,80,120,160,200], [200,160,120,80,40,0], -1.),
    ([0,40,80,120,160,200], [120]*6, 0.),
    ([0,40,80], [0,80,80], .8),  # observed squared error 1, expected 5
    ([0,200], [200,0], -1.),
    ([120,120], [120,120], None),
    ([120,120], [80,80], 0.),
])
def test_qwk_known_mathematical_examples(truth, prediction, expected):
    result = score_metrics(np.repeat(np.array(truth)[:,None], 5, axis=1),
                           np.repeat(np.array(prediction)[:,None], 5, axis=1))
    for metric in [*result['competencies'].values(), result['total'], result['macro']]:
        if expected is None:
            assert metric['qwk'] is None
        else:
            assert metric['qwk'] == pytest.approx(expected)


def test_qwk_matches_independent_expected_disagreement_with_missing_classes():
    rng = np.random.default_rng(17)
    truth = rng.choice([0,80,160,200], (81,5))
    prediction = rng.choice([40,80,120,200], (81,5))
    result = score_metrics(truth, prediction)
    for i in range(5):
        observed = np.mean((truth[:,i]-prediction[:,i])**2)
        expected = np.mean((truth[:,i,None]-prediction[:,i][None,:])**2)
        assert result['competencies'][f'c{i+1}']['qwk'] == pytest.approx(1-observed/expected)


def test_competencies_are_not_flattened_or_reordered():
    truth = np.array([[0,40,80,120,160],[200,160,120,80,40]])
    prediction = truth.copy(); prediction[:,2] = prediction[::-1,2]
    metrics = score_metrics(truth,prediction)
    assert [metrics['competencies'][f'c{i}']['qwk'] for i in range(1,6)] == [1.,1.,-1.,1.,1.]
