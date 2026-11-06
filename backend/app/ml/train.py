"""Fine-tune local BERTimbau; choose checkpoints using development QWK only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from .dataset import file_hash,load_records,write_json,verify_prepared
from .baseline import selection_key
from .metrics import score_metrics
from .transformer import (EssayScorer, encode_cached, batch_from_cached, select_cached,
                          load_checkpoint, predict_texts, save_checkpoint)
from .checkpoints import BackupManager, CheckpointError


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False


def train_transformer(data_dir,base_model,output,objective='ordinal',epochs=8,batch_size=2,
                      learning_rate=2e-5,weight_decay=.01,patience=2,seed=42,max_length=512,
                      overlap=64,device='auto',resume=False,chunk_batch_size=8,
                      max_train=None,max_development=None,training_mode='full',
                      backup_interval_seconds=1800,max_backups=2,backup_root=None,
                      run_id=None,resume_latest=False,resume_from=None):
    if epochs<1 or batch_size<1 or patience<1 or learning_rate<=0:
        raise ValueError('epochs, batch_size, patience and learning_rate must be positive')
    if training_mode not in ('full','frozen'):
        raise ValueError('training_mode must be full or frozen')
    data_dir,base_model,output=Path(data_dir),Path(base_model),Path(output)
    if not base_model.is_dir():
        raise FileNotFoundError('BERTimbau local ausente. Execute o download explícito antes do treino.')
    set_seed(seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    verify_prepared(data_dir)
    training=load_records(data_dir/'training.jsonl');development=load_records(data_dir/'development.jsonl')
    if not training or not development: raise ValueError('Empty training or development split')
    if max_train is not None:
        if max_train < 1: raise ValueError('max_train must be positive')
        training=training[:max_train]
    if max_development is not None:
        if max_development < 1: raise ValueError('max_development must be positive')
        development=development[:max_development]
    provenance={split:file_hash(data_dir/f'{split}.jsonl') for split in ('training','development')}
    tokenizer=AutoTokenizer.from_pretrained(base_model,local_files_only=True)
    encoder=AutoModel.from_pretrained(base_model,local_files_only=True)
    if max_length>encoder.config.max_position_embeddings:
        raise ValueError('max_length exceeds encoder position capacity')
    # Validate chunk options before the expensive training loop.
    from .transformer import chunk_token_ids
    chunk_token_ids([1],max_length-tokenizer.num_special_tokens_to_add(False),overlap)
    model=EssayScorer(encoder,objective,chunk_batch_size,training_mode == 'frozen').to(device)
    hp={'objective':objective,'epochs':epochs,'batch_size':batch_size,'learning_rate':learning_rate,
        'weight_decay':weight_decay,'patience':patience,'seed':seed,'max_length':max_length,
        'overlap':overlap,'chunk_batch_size':chunk_batch_size,'training_mode':training_mode,
        'max_train':max_train,'max_development':max_development}
    fingerprint=hashlib.sha256(json.dumps({'data':provenance,'hp':hp,'base_config':encoder.config.to_dict()},sort_keys=True).encode()).hexdigest()[:12]
    version=f'bertimbau-{objective}-{fingerprint}'
    limited=max_train is not None or max_development is not None
    metadata={'schema_version':1,'version':version,'trained':not limited,'engineering_smoke':limited,
        'created_at':datetime.now(timezone.utc).isoformat(),'base_model':str(base_model.resolve()),
        'hyperparameters':hp,'data_sha256':provenance,'seed':seed,'max_length':max_length,'overlap':overlap,
        'training_scope':'limited smoke subset' if limited else 'prepared training split',
        'selection_split':'development','selection_metric':'macro_qwk_then_macro_mae','test_evaluated':False,
        'train_count':len(training),'development_count':len(development),'device':str(device),
        'torch_version':torch.__version__}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=learning_rate,weight_decay=weight_decay)
    total_steps=((len(training)+batch_size-1)//batch_size)*epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*total_steps),total_steps)
    history=[];start_epoch=0;next_batch=0;resume_indices=None;global_step=0;best=None;stale=0
    output.mkdir(parents=True,exist_ok=True)
    if backup_interval_seconds <= 0 or max_backups < 1:
        raise ValueError('backup_interval_seconds and max_backups must be positive')
    backup_root = Path(backup_root) if backup_root else output.parent/'training-backups'
    if resume_latest and run_id is None and backup_root.exists():
        candidates = []
        for candidate in backup_root.iterdir():
            if candidate.is_dir():
                try:
                    manager = BackupManager(backup_root,candidate.name,max_backups)
                    candidates.append((manager.latest().stat().st_mtime_ns,candidate.name))
                except (CheckpointError, OSError, ValueError):
                    continue
        if candidates:
            run_id=max(candidates)[1]
    run_id = run_id or f'run-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}-{version}'
    backups = BackupManager(backup_root,run_id,max_backups)
    cache_dir=output/'token-cache'
    cached_training=encode_cached(tokenizer,training,max_length,overlap,cache_dir)
    last=output/'checkpoint-last'
    resume_path = Path(resume_from) if resume_from else (backups.latest()/'model' if resume_latest else last)
    if resume or resume_latest or resume_from:
        if not resume_path.exists():
            raise ValueError(f'Resume checkpoint not found: {resume_path}')
        resumed,saved=load_checkpoint(resume_path,device)
        if saved['version']!=version:
            raise ValueError('Resume data/model/hyperparameters differ from the saved checkpoint')
        model.load_state_dict(resumed.state_dict())
        state_path = resume_path.parent/'trainer_state.pt' if resume_path.name == 'model' else last/'trainer.pt'
        state=torch.load(state_path,map_location='cpu',weights_only=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start_epoch=state['epoch'];next_batch=state.get('next_batch',0)
        resume_indices=state.get('indices')
        global_step=state.get('global_step',0);best=state['best'];stale=state['stale'];history=state['history']
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        if device.startswith('cuda') and state.get('cuda_rng'): torch.cuda.set_rng_state_all(state['cuda_rng'])
    next_backup=time.monotonic()+backup_interval_seconds
    started=time.monotonic()
    def save_backup(epoch, next_batch, indices):
        state_payload={'epoch':epoch,'next_batch':next_batch,'global_step':global_step,
                       'indices':indices,'best':best,'stale':stale,'history':history,
                       'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
                       'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}
        def write_backup(directory):
            save_checkpoint(model,directory/'model',metadata|{'epoch':epoch,'run_id':run_id},tokenizer)
            torch.save(optimizer.state_dict(),directory/'optimizer.pt')
            torch.save(scheduler.state_dict(),directory/'scheduler.pt')
            torch.save(state_payload,directory/'trainer_state.pt')
        return backups.save(write_backup)
    for epoch in range(start_epoch,epochs):
        model.train()
        if training_mode == 'frozen':
            model.encoder.eval()
        indices=resume_indices if resume_indices is not None else list(range(len(training)))
        if resume_indices is None:
            random.shuffle(indices)
        resume_indices=None
        loss_sum=0.;steps=0
        batch_start=next_batch if epoch == start_epoch else 0
        for start in range(batch_start,len(indices),batch_size):
            batch_indices=indices[start:start+batch_size]
            batch=[training[i] for i in batch_indices]
            cached_batch=select_cached(cached_training,batch_indices)
            ids,mask,owners=batch_from_cached(tokenizer,cached_batch,0,len(cached_batch))
            targets=torch.tensor([r['scores'] for r in batch],device=device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(ids.to(device),mask.to(device),owners.to(device),len(batch))
            loss=model.loss(logits,targets)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step();loss_sum+=float(loss.detach());steps+=1
            global_step += 1
            next_batch = start + batch_size
            if time.monotonic() >= next_backup:
                backup=save_backup(epoch,next_batch,indices)
                print(json.dumps({'backup':str(backup),'retained':[str(p) for p in backups.valid_backups()]},default=str),flush=True)
                next_backup=time.monotonic()+backup_interval_seconds
        predictions=predict_texts(model,tokenizer,[r['text'] for r in development],max_length,overlap,batch_size)
        metrics=score_metrics([r['scores'] for r in development],predictions)
        entry={'epoch':epoch+1,'train_loss':loss_sum/steps,'validation_metrics':metrics,
               'elapsed_seconds':time.monotonic()-started,'learning_rate':scheduler.get_last_lr()[0]}
        history.append(entry)
        if best is None or selection_key(metrics)<selection_key(best):
            best=metrics;stale=0
            save_checkpoint(model,output,metadata|{'best_epoch':epoch+1,'validation_metrics':metrics},tokenizer)
        else: stale+=1
        save_checkpoint(model,last,metadata|{'epoch':epoch+1},tokenizer)
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,
                    'best':best,'stale':stale,'history':history,'python_rng':random.getstate(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},last/'trainer.pt')
        write_json(output/'training_log.json',{'version':version,'hyperparameters':hp,'epochs':history,
            'early_stopped':stale>=patience,'selection_split':'development','test_evaluated':False})
        print(json.dumps({'epoch':epoch+1,'loss':entry['train_loss'],'development':metrics['macro']},allow_nan=False),flush=True)
        if time.monotonic() >= next_backup:
            backup=save_backup(epoch+1,0,[])
            print(json.dumps({'backup':str(backup),'retained':[str(p) for p in backups.valid_backups()]},default=str),flush=True)
            next_backup=time.monotonic()+backup_interval_seconds
        next_batch=0
        if stale>=patience: break
    return {'version':version,'validation_metrics':best,'epochs_completed':len(history),'test_evaluated':False,
            'backup_run_id':run_id,'backups':[str(p) for p in backups.valid_backups()]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('datasets/prepared'))
    p.add_argument('--base-model',type=Path,default=Path('artifacts/models/bertimbau'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--objective',choices=['ordinal','regression'],default='ordinal')
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=2e-5);p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--patience',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-length',type=int,default=512);p.add_argument('--overlap',type=int,default=64)
    p.add_argument('--device',default='auto');p.add_argument('--resume',action='store_true')
    p.add_argument('--chunk-batch-size',type=int,default=8)
    p.add_argument('--max-train',type=int,default=None)
    p.add_argument('--max-development',type=int,default=None)
    p.add_argument('--training-mode',choices=['full','frozen'],default='full')
    p.add_argument('--backup-interval-seconds',type=float,default=1800)
    p.add_argument('--max-backups',type=int,default=2)
    p.add_argument('--backup-root',type=Path,default=None)
    p.add_argument('--run-id',default=None)
    p.add_argument('--resume-latest',action='store_true')
    p.add_argument('--resume-from',type=Path,default=None)
    args=vars(p.parse_args());args['data_dir']=args.pop('data')
    print(json.dumps(train_transformer(**args),indent=2))

if __name__=='__main__': main()
