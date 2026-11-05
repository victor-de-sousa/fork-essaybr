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
