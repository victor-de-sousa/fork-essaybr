import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.core.config import Settings
from app.services.ocr_service import OCRService, segment_lines
from app.services.ocr_metrics import transcription_metrics


def test_segment_two_lines_without_spelling_correction():
    image = Image.new('RGB', (300, 150), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((15,20,230,28), fill='black')
    draw.rectangle((15,60,200,68), fill='black')
    lines = segment_lines(image)
    assert len(lines) == 2
    assert lines[1][1] > lines[0][1]


def test_missing_ocr_weights_fail(tmp_path):
    service = OCRService(Settings(root_dir=tmp_path))
    assert not service.status()['available']
    with pytest.raises(RuntimeError, match='OCR'):
        service.transcribe([])


def test_cer_wer_keep_diacritics_and_errors():
    result = transcription_metrics(['ação boa'], ['acao boa'])
    assert result['cer'] == pytest.approx(2/8)
    assert result['wer'] == 0.5
    assert transcription_metrics(['Olá.'], ['Olá.'])['cer'] == 0
