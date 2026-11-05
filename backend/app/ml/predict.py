"""Lazy local artifact loading. A missing model never produces substitute grades."""
from __future__ import annotations
import json
from pathlib import Path
from .metrics import quantize


class ModelUnavailableError(RuntimeError):
    pass


class Predictor:
    def __init__(self, model_dir: Path):
        self.model_dir=Path(model_dir)
        self._model=None
        self._tokenizer=None
        self._loaded_version=None

    def status(self):
        path=self.model_dir/'metadata.json'
        if not path.exists():
            return {'trained':False,'available':False,'kind':None,'version':None,'message':'Modelo supervisionado ausente. Execute o treinamento local.'}
        try:
            metadata=json.loads(path.read_text(encoding='utf-8'))
            artifact=self.model_dir/('baseline.joblib' if metadata.get('kind')=='tfidf_ridge' else 'model.safetensors')
            available=bool(metadata.get('trained')) and artifact.exists() and metadata.get('kind') in ('tfidf_ridge','transformer_ordinal','transformer_regression')
            return {**metadata,'available':available,'trained':available,
                    'message':'Modelo supervisionado local disponível.' if available else 'Artefato treinado incompleto ou apenas teste de engenharia.'}
        except (ValueError,OSError) as exc:
            return {'trained':False,'available':False,'kind':None,'version':None,'message':f'Metadados inválidos: {exc}'}

    def predict_raw(self,texts: list[str]):
        if not texts or any(not isinstance(text,str) or not text.strip() for text in texts):
            raise ValueError('Forneça uma redação não vazia.')
        metadata=self.status()
        if not metadata['trained']:
            raise ModelUnavailableError(metadata['message'])
        if self._model is None or self._loaded_version!=metadata['version']:
            try:
                if metadata['kind']=='tfidf_ridge':
                    import joblib
                    from .dataset import file_hash
                    if metadata.get('artifact_sha256') != file_hash(self.model_dir/'baseline.joblib'):
                        raise ModelUnavailableError('Checksum do modelo não corresponde aos metadados.')
                    self._model=joblib.load(self.model_dir/'baseline.joblib')
                else:
                    from .transformer import load_checkpoint
                    from transformers import AutoTokenizer
                    self._model,_=load_checkpoint(self.model_dir)
                    self._tokenizer=AutoTokenizer.from_pretrained(self.model_dir/'tokenizer',local_files_only=True)
                self._loaded_version=metadata['version']
            except (OSError,ValueError,KeyError) as exc:
                raise ModelUnavailableError(f'Não foi possível carregar o modelo local: {exc}') from exc
        if metadata['kind']=='tfidf_ridge':
            return self._model.predict(texts)
        from .transformer import predict_texts
        return predict_texts(self._model,self._tokenizer,texts,
                             metadata.get('max_length',512),metadata.get('overlap',64))

    def predict(self,text: str):
        values=quantize(self.predict_raw([text]))[0]
        metadata=self.status()
        return {'scores':{f'c{i+1}':int(value) for i,value in enumerate(values)},
                'model_info':{k:metadata[k] for k in ('version','trained','kind')}}
