"""Bounded CPU benchmark on train/development only; never trains a full model."""
import argparse
import json
import os
import platform
from pathlib import Path
import time
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from .dataset import load_records, file_hash, write_json, verify_prepared
from .transformer import (load_checkpoint, encode_cached, encode_essays,
                          select_cached, batch_from_cached, decode_predictions)


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('artifacts/models/bertimbau-frozen'))
    p.add_argument('--data', type=Path, default=Path('datasets/prepared'))
    p.add_argument('--train-samples', type=int, default=32)
    p.add_argument('--dev-samples', type=int, default=16)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--with-cache', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    verify_prepared(args.data)
    records = {s: load_records(args.data / f'{s}.jsonl') for s in ('training', 'development')}
    rng = np.random.default_rng(42)
    indices = {s: rng.choice(len(records[s]), n, replace=False).tolist()
               for s, n in [('training', args.train_samples), ('development', args.dev_samples)]}
    sample = {s: [records[s][i] for i in ix] for s, ix in indices.items()}
    model, metadata = load_checkpoint(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'tokenizer', local_files_only=True)
    length, overlap = metadata['max_length'], metadata['overlap']
    report = {'test_used': False, 'seed': 42, 'sample_ids': {s: [r['id'] for r in rows] for s, rows in sample.items()},
              'platform': platform.platform(), 'cpu': next(line.strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'threads': args.threads, 'interop_threads': 1, 'affinity': sorted(os.sched_getaffinity(0)),
              'model_sha256': file_hash(args.model / 'model.safetensors'),
              'data_sha256': {s: file_hash(args.data / f'{s}.jsonl') for s in records},
              'batch_size': 4, 'chunk_batch_size': model.chunk_batch_size, 'max_length': length, 'overlap': overlap}
    tokens, report['tokenize_training_seconds'] = timed(lambda: encode_cached(tokenizer, records['training'], length, overlap, args.output / 'tokens'))
    _, report['token_cache_load_seconds'] = timed(lambda: encode_cached(tokenizer, records['training'], length, overlap, args.output / 'tokens'))
    report['training_chunks'] = len(tokens)
    batches = []
    selection, padding = 0., 0.
    for start in range(0, len(sample['training']), 4):
        chosen, elapsed = timed(lambda: select_cached(tokens, indices['training'][start:start+4]))
        selection += elapsed
        batch, elapsed = timed(lambda: batch_from_cached(tokenizer, chosen, 0, len(chosen)))
        padding += elapsed
        batches.append(batch)
    report['selection_seconds'], report['padding_seconds'] = selection, padding
    report['sample_chunks'] = sum(len(batch[0]) for batch in batches)
    report['padding_fraction'] = 1 - sum(int(b[1].sum()) for b in batches) / sum(b[1].numel() for b in batches)
    report['thread_sweep'] = {}
    ids, mask, owners = batches[0]
    for threads in [1, 2, 4, 8, 10, 12]:
        torch.set_num_threads(threads)
        with torch.no_grad():
            model(ids, mask, owners, 4)
            durations = [timed(lambda: model(ids, mask, owners, 4))[1] for _ in range(2)]
        report['thread_sweep'][str(threads)] = durations
        print(json.dumps({'threads': threads, 'forward_seconds': durations}), flush=True)
    torch.set_num_threads(args.threads)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith('encoder.')}
    targets = torch.tensor([r['scores'] for r in sample['training']])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    totals = dict(encoder=0., head_forward=0., forward=0., loss=0., backward=0., optimizer=0.)
    clock = {}
    def before(name):
        def hook(module, inputs):
            clock[name] = time.perf_counter()
        return hook
    def after(name):
        def hook(module, inputs, output):
            totals[name] += time.perf_counter() - clock[name]
        return hook
    hooks = [model.encoder.register_forward_pre_hook(before('encoder')), model.encoder.register_forward_hook(after('encoder')),
             model.head.register_forward_pre_hook(before('head_forward')), model.head.register_forward_hook(after('head_forward'))]
    outputs, gradients = [], []
    for j, (ids, mask, owners) in enumerate(batches):
        model.train(); model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        logits, elapsed = timed(lambda: model(ids, mask, owners, len(targets[j*4:j*4+4])))
        totals['forward'] += elapsed
        outputs.append(logits.detach().clone())
        loss, elapsed = timed(lambda: model.loss(logits, targets[j*4:j*4+4]))
        totals['loss'] += elapsed
        _, elapsed = timed(loss.backward); totals['backward'] += elapsed
        gradients.append(model.head.weight.grad.clone())
        def step():
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        _, elapsed = timed(step); totals['optimizer'] += elapsed
    for hook in hooks:
        hook.remove()
    report['online_training_seconds'] = totals
    report['validation'] = {'tokenization_padding': 0., 'forward': 0., 'chunks': 0}
    dev_outputs=[]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample['development']), 4):
            rows = sample['development'][start:start+4]
            (ids, mask, owners), elapsed = timed(lambda: encode_essays(tokenizer, [r['text'] for r in rows], length, overlap))
            report['validation']['tokenization_padding'] += elapsed
            report['validation']['chunks'] += len(ids)
            logits, elapsed = timed(lambda: model(ids, mask, owners, len(rows)))
            dev_outputs.append(logits.detach().clone())
            report['validation']['forward'] += elapsed
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.no_grad():
            model(ids, mask, owners, len(rows))
    (args.output / 'profile.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=18))
    if args.with_cache:
        from .features import cached_features
        model.load_state_dict(initial, strict=False)
        features, report['feature_cache_cold_seconds'] = timed(lambda: cached_features(model, tokenizer, sample['training'], length, overlap, args.output / 'features', batch_size=4))
        dev_features, report['dev_feature_cache_cold_seconds'] = timed(lambda: cached_features(model, tokenizer, sample['development'], length, overlap, args.output / 'features', batch_size=4))
        reloaded, report['feature_cache_warm_seconds'] = timed(lambda: cached_features(model, tokenizer, sample['training'], length, overlap, args.output / 'features', batch_size=4))
        assert torch.equal(features, reloaded)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        cached_outputs, cached_gradients = [], []
        def epoch():
            for start in range(0, len(features), 4):
                optimizer.zero_grad(set_to_none=True)
                logits = model.forward_features(features[start:start+4])
                loss = model.loss(logits, targets[start:start+4]); loss.backward()
                cached_outputs.append(logits.detach().clone())
                cached_gradients.append(model.head.weight.grad.clone())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        _, report['cached_train_seconds'] = timed(epoch)
        with torch.no_grad():
            dev_logits, report['cached_validation_seconds'] = timed(lambda: model.forward_features(dev_features))
        report['dev_logits_max_abs_error']=float((torch.cat(dev_outputs)-dev_logits).abs().max())
        report['logits_max_abs_error'] = max(float((a-b).abs().max()) for a, b in zip(outputs, cached_outputs))
        report['head_gradients_max_abs_error'] = max(float((a-b).abs().max()) for a, b in zip(gradients, cached_gradients))
        report['decoded_identical'] = all(torch.equal(decode_predictions(a, 'ordinal'), decode_predictions(b, 'ordinal')) for a, b in zip(outputs, cached_outputs))
        assert report['logits_max_abs_error'] < 1e-4 and report['head_gradients_max_abs_error'] < 1e-5 and report['decoded_identical']
        assert report['dev_logits_max_abs_error'] < 1e-4
        report['features_bytes']=features.numel()*features.element_size()
        report['head_grad_norms']=[float(g.norm()) for g in gradients]
        report['sample_head_values_std']=model.head(features).detach().std(0).tolist()
        full_targets = torch.tensor([r['scores'] for r in records['training']])
        # Timing-only shape simulation; no claims about quality or training on full corpus.
        expanded = features.repeat((len(full_targets)+len(features)-1)//len(features), 1)[:len(full_targets)]
        def shape_epoch():
            for start in range(0, len(full_targets), 4):
                optimizer.zero_grad(set_to_none=True)
                model.loss(model.forward_features(expanded[start:start+4]), full_targets[start:start+4]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        _, report['head_only_3194_shape_benchmark_seconds'] = timed(shape_epoch)
        # Save only diagnostics; these timing-only updates are never an artifact/model.
    write_json(args.output / 'results.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
