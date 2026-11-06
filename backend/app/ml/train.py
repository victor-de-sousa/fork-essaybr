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
from .transformer import EssayScorer,encode_essays,load_checkpoint,predict_texts,save_checkpoint


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False


def train_transformer(data_dir,base_model,output,objective='ordinal',epochs=8,batch_size=2,
                      learning_rate=2e-5,weight_decay=.01,patience=2,seed=42,max_length=512,
                      overlap=64,device='auto',resume=False,chunk_batch_size=8,
                      max_train=None,max_development=None):
    if epochs<1 or batch_size<1 or patience<1 or learning_rate<=0:
        raise ValueError('epochs, batch_size, patience and learning_rate must be positive')
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
    model=EssayScorer(encoder,objective,chunk_batch_size).to(device)
    hp={'objective':objective,'epochs':epochs,'batch_size':batch_size,'learning_rate':learning_rate,
        'weight_decay':weight_decay,'patience':patience,'seed':seed,'max_length':max_length,
        'overlap':overlap,'chunk_batch_size':chunk_batch_size}
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
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    total_steps=((len(training)+batch_size-1)//batch_size)*epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*total_steps),total_steps)
    history=[];start_epoch=0;best=None;stale=0
    last=output/'checkpoint-last'
    if resume:
        resumed,saved=load_checkpoint(last,device)
        if saved['version']!=version:
            raise ValueError('Resume data/model/hyperparameters differ from the saved checkpoint')
        model.load_state_dict(resumed.state_dict())
        state=torch.load(last/'trainer.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start_epoch=state['epoch']+1;best=state['best'];stale=state['stale'];history=state['history']
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        if device.startswith('cuda') and state.get('cuda_rng'): torch.cuda.set_rng_state_all(state['cuda_rng'])
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    for epoch in range(start_epoch,epochs):
        model.train();indices=list(range(len(training)));random.shuffle(indices);loss_sum=0.;steps=0
        for start in range(0,len(indices),batch_size):
            batch=[training[i] for i in indices[start:start+batch_size]]
            ids,mask,owners=encode_essays(tokenizer,[r['text'] for r in batch],max_length,overlap)
            targets=torch.tensor([r['scores'] for r in batch],device=device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(ids.to(device),mask.to(device),owners.to(device),len(batch))
            loss=model.loss(logits,targets)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step();loss_sum+=float(loss.detach());steps+=1
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
        if stale>=patience: break
    return {'version':version,'validation_metrics':best,'epochs_completed':len(history),'test_evaluated':False}


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
    args=vars(p.parse_args());args['data_dir']=args.pop('data')
    print(json.dumps(train_transformer(**args),indent=2))

if __name__=='__main__': main()
