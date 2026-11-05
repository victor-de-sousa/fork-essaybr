"""BERT essay scoring over every token: chunk encoding and essay-level aggregation."""
from __future__ import annotations
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from .dataset import write_json


def chunk_token_ids(ids, content_length=510, overlap=64):
    if content_length <= 0 or not 0 <= overlap < content_length:
        raise ValueError('Require content_length > overlap >= 0')
    if not ids:
        return [[]]
    chunks=[]
    for start in range(0,len(ids),content_length-overlap):
        chunk=ids[start:start+content_length]
        chunks.append(chunk)
        if start+content_length >= len(ids):
            break
    return chunks


def encode_essays(tokenizer,texts,max_length=512,overlap=64):
    content_length=max_length-tokenizer.num_special_tokens_to_add(pair=False)
    chunks,owners=[],[]
    for owner,text in enumerate(texts):
        # Explicitly disable truncation: all tokens participate in a chunk.
        ids=tokenizer.encode(text,add_special_tokens=False,truncation=False)
        for chunk in chunk_token_ids(ids,content_length,overlap):
            chunks.append(tokenizer.prepare_for_model(chunk,add_special_tokens=True,
                truncation=False,padding=False,return_attention_mask=True))
            owners.append(owner)
    batch=tokenizer.pad(chunks,padding=True,return_tensors='pt')
    return batch['input_ids'],batch['attention_mask'],torch.tensor(owners,dtype=torch.long)


def ordinal_targets(scores):
    return (scores.unsqueeze(-1)/40 > torch.arange(5,device=scores.device)).float()


def decode_predictions(logits,objective):
    if objective=='ordinal':
        # Monotone cumulative logits give the median grade, minimizing ordinal absolute loss.
        return (logits>=0).sum(-1)*40
    if objective=='regression':
        return torch.sigmoid(logits)*200
    raise ValueError(f'Unknown objective {objective}')


class EssayScorer(nn.Module):
    def __init__(self,encoder,objective='ordinal',chunk_batch_size=8):
        super().__init__()
        if objective not in ('ordinal','regression'):
            raise ValueError('objective must be ordinal or regression')
        self.encoder=encoder
        self.objective=objective
        self.chunk_batch_size=chunk_batch_size
        self.head=nn.Linear(encoder.config.hidden_size,5)
        if objective=='ordinal':
            self.first_threshold=nn.Parameter(torch.full((5,1),-2.))
            self.threshold_increments=nn.Parameter(torch.ones(5,4))

    def forward(self,input_ids,attention_mask,owners,essay_count):
        pooled=[]
        for start in range(0,len(input_ids),self.chunk_batch_size):
            result=self.encoder(input_ids=input_ids[start:start+self.chunk_batch_size],
                                attention_mask=attention_mask[start:start+self.chunk_batch_size])
            pooled.append(result.last_hidden_state[:,0])
        embeddings=torch.cat(pooled)
        # Length-weighted pooling includes the final partial chunk instead of discarding it.
        weights=attention_mask.sum(1).to(embeddings.dtype)
        sums=embeddings.new_zeros((essay_count,embeddings.shape[-1])).index_add(0,owners,embeddings*weights[:,None])
        totals=embeddings.new_zeros(essay_count).index_add(0,owners,weights)
        features=sums/totals[:,None].clamp_min(1)
        values=self.head(features)
        if self.objective=='regression':
            return values
        thresholds=torch.cat((self.first_threshold,self.first_threshold+F.softplus(self.threshold_increments).cumsum(-1)),dim=-1)
        return values.unsqueeze(-1)-thresholds.unsqueeze(0)

    def loss(self,logits,scores):
        if self.objective=='ordinal':
            return F.binary_cross_entropy_with_logits(logits,ordinal_targets(scores))
        return F.mse_loss(torch.sigmoid(logits),scores.float()/200)


def save_checkpoint(model,path: Path,metadata: dict,tokenizer=None):
    from safetensors.torch import save_file
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    model.encoder.config.save_pretrained(path/'encoder')
    save_file({key:value.detach().cpu().contiguous() for key,value in model.state_dict().items()},str(path/'model.safetensors'))
    if tokenizer is not None:
        tokenizer.save_pretrained(path/'tokenizer')
    write_json(path/'metadata.json',metadata | {'objective':model.objective,'kind':f'transformer_{model.objective}',
        'chunk_batch_size':model.chunk_batch_size})


def load_checkpoint(path: Path,device='cpu'):
    from safetensors.torch import load_file
    from transformers import AutoConfig,AutoModel
    path=Path(path)
    metadata=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    config=AutoConfig.from_pretrained(path/'encoder',local_files_only=True)
    model=EssayScorer(AutoModel.from_config(config),metadata['objective'],metadata.get('chunk_batch_size',8))
    model.load_state_dict(load_file(str(path/'model.safetensors'),device='cpu'))
    model.to(device);model.eval()
    return model,metadata


def predict_texts(model,tokenizer,texts,max_length=512,overlap=64,batch_size=4):
    import numpy as np
    model.eval();device=next(model.parameters()).device;predictions=[]
    with torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            batch=texts[start:start+batch_size]
            ids,mask,owners=encode_essays(tokenizer,batch,max_length,overlap)
            values=model(ids.to(device),mask.to(device),owners.to(device),len(batch))
            predictions.append(decode_predictions(values,model.objective).cpu().numpy())
    return np.concatenate(predictions)
