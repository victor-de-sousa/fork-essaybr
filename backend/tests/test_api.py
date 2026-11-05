import io
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def photo(color='white', fmt='PNG'):
    stream = io.BytesIO()
    Image.new('RGB', (100, 150), color).save(stream, fmt)
    return stream.getvalue()


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(root_dir=tmp_path))
    with TestClient(app) as c:
        yield c


def upload(client):
    response = client.post('/api/essays/upload', files=[('files', ('../../a.png', photo(), 'image/png')), ('files', ('b.png', photo('gray'), 'image/png'))], data={'theme': 'Educação'})
    assert response.status_code == 201, response.text
    return response.json()


def test_upload_order_images_history_delete(client):
    essay = upload(client)
    assert [i['position'] for i in essay['images']] == [1, 2]
    assert client.get(essay['images'][0]['url']).status_code == 200
    assert len(client.get('/api/essays').json()) == 1
    folder = client.app.state.settings.images_dir / essay['id']
    assert [p.name for p in sorted(folder.iterdir())] == ['page_01.png', 'page_02.png']
    assert client.delete('/api/essays/' + essay['id']).status_code == 204
    assert not folder.exists()
    assert client.get('/api/essays/' + essay['id']).status_code == 404


def test_invalid_upload_atomic(client):
    r = client.post('/api/essays/upload', files=[('files', ('a.png', photo(), 'image/png')), ('files', ('b.png', b'bad', 'image/png'))])
    assert r.status_code == 422
    assert r.json()['error']['message']
    assert client.get('/api/essays').json() == []
    assert not list(client.app.state.settings.images_dir.iterdir())


def test_requires_confirmed_transcription_and_missing_models_clear(client):
    essay = upload(client); url = '/api/essays/' + essay['id']
    assert client.post(url + '/correct').status_code == 409
    r = client.patch(url + '/transcription', json={'text':'A educação é um direito.','confirmed':True})
    assert r.status_code == 200
    assert r.json()['confirmed'] is True
    response = client.post(url + '/correct')
    assert response.status_code == 503
    assert client.get(url).json()['status'] == 'failed'


def test_mock_ocr_preserves_original_and_requires_confirmation(client):
    essay = upload(client); url = '/api/essays/' + essay['id']
    class OCR:
        def transcribe(self, paths):
            assert len(paths) == 2
            return {'text': 'Os aluno estuda.\n\n[ilegível]', 'warnings': ['Trecho incerto.'], 'pages': []}
    client.app.state.ocr = OCR()
    r = client.post(url + '/transcribe')
    assert r.status_code == 200
    assert r.json()['transcription'].startswith('Os aluno estuda.')
    assert not r.json()['confirmed']
    assert client.post(url + '/correct').status_code == 409


def test_persistence_and_score_validation(tmp_path):
    settings = Settings(root_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        essay = upload(c)
    with TestClient(create_app(settings)) as c:
        assert c.get('/api/essays/' + essay['id']).status_code == 200
        bad = {'scores':dict(c1=42,c2=80,c3=80,c4=80,c5=80),'notes':'x'}
        assert c.post('/api/essays/' + essay['id'] + '/reference',json=bad).status_code == 422


def test_local_status_does_not_claim_training(client):
    r = client.get('/api/model/status')
    assert r.status_code == 200, r.text
    assert not r.json()['scoring'].get('trained', False)
