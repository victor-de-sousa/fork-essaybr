"""Persistent, verified FP32 essay features for a deterministic frozen encoder."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch
import transformers
from .transformer import encode_essays

POOLING_VERSION = 'cls-length-weighted-including-specials-and-overlap-v1'


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def encoder_digest(encoder):
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor_digest(tensor).encode())
    return digest.hexdigest()


def feature_identity(model, tokenizer, records, max_length, overlap):
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    return {
        'schema': 1, 'pooling': POOLING_VERSION,
        'encoder_sha256': encoder_digest(model.encoder),
        'encoder_config': model.encoder.config.to_dict(),
        'tokenizer': backend.to_str() if backend is not None else tokenizer.get_vocab(),
        'special_tokens': tokenizer.special_tokens_map,
        'tokenizer_options': tokenizer.init_kwargs,
        'padding_side': tokenizer.padding_side,
        'records': [(r['id'], r['text']) for r in records],
        'max_length': max_length, 'overlap': overlap,
        'torch': str(torch.__version__), 'transformers': transformers.__version__,
        'device_type': next(model.encoder.parameters()).device.type,
        'dtype': str(next(model.encoder.parameters()).dtype),
    }


def cached_features(model, tokenizer, records, max_length, overlap, cache_dir, batch_size=4):
    """Cache only features; labels always come from the current verified split.

    Batch size does not alter right-padded BERT's mathematical result. FP32
    roundoff across batching strategies is covered by parity tests/benchmarks.
    no_grad (not inference_mode) permits autograd to save features for the head.
    """
    if not model.frozen or any(p.requires_grad for p in model.encoder.parameters()):
        raise ValueError('Feature cache requires a fully frozen encoder')
    if tokenizer.padding_side != 'right':
        raise ValueError('CLS feature cache requires right padding')
    if next(model.encoder.parameters()).dtype != torch.float32:
        raise ValueError('Feature cache currently requires FP32 encoder weights')
    if batch_size < 1 or not records:
        raise ValueError('Require nonempty records and positive batch_size')
    identity = feature_identity(model, tokenizer, records, max_length, overlap)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{key}.pt'
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        features = payload['features']
        if (payload.get('key') != key or features.shape != (len(records), model.head.in_features)
                or features.dtype != torch.float32 or not torch.isfinite(features).all()
                or payload.get('sha256') != tensor_digest(features)):
            raise ValueError(f'Feature cache integrity failure: {path}')
        return features
    device = next(model.encoder.parameters()).device
    was_training = model.training
    values = []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                rows = records[start:start+batch_size]
                ids, mask, owners = encode_essays(tokenizer, [r['text'] for r in rows], max_length, overlap)
                values.append(model.encode_features(ids.to(device), mask.to(device), owners.to(device), len(rows)).float().cpu())
        features = torch.cat(values).contiguous()
    finally:
        model.train(was_training)
    if not torch.isfinite(features).all():
        raise ValueError('Non-finite encoder features')
    payload = {'features': features, 'key': key, 'sha256': tensor_digest(features),
               'pooling': POOLING_VERSION, 'encoder_sha256': identity['encoder_sha256']}
    # Unique temporary file prevents concurrent writers from sharing a .tmp path.
    fd, temporary = tempfile.mkstemp(prefix=f'.{key}-', suffix='.tmp', dir=cache_dir)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return features
