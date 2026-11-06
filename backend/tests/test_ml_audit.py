"""Regression tests for frozen extraction, content identity and epoch resumption."""
import json
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from app.ml.transformer import EssayScorer, encode_essays, ordinal_targets, decode_predictions


@pytest.fixture
def tiny(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = tmp_path/'base'; base.mkdir()
    (base/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\nd\ne\n')
    tokenizer = BertTokenizerFast(vocab_file=str(base/'vocab.txt'), do_lower_case=False)
    tokenizer.save_pretrained(base)
    encoder = BertModel(BertConfig(vocab_size=10, hidden_size=12, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=20,
                                  hidden_dropout_prob=.5, attention_probs_dropout_prob=.5))
    encoder.save_pretrained(base)
    return base, tokenizer, EssayScorer(encoder, frozen=True, chunk_batch_size=2)


def test_frozen_train_disables_encoder_dropout(tiny):
    _, tokenizer, model = tiny
    model.train()
    batch = encode_essays(tokenizer, ['a b c d e a b c d e', 'c a'], 8, 2)
    first = model(*batch, 2)
    second = model(*batch, 2)
    assert torch.equal(first, second)
    assert model.training and not model.encoder.training


def test_ordinal_round_trip_all_grades_all_competencies():
    scores = torch.tensor([[0,40,80,120,160], [200,160,120,80,40]])
    logits = ordinal_targets(scores)*20-10
    assert torch.equal(decode_predictions(logits,'ordinal'),scores)
    # Exact zero is the documented upper-median tie convention.
    assert decode_predictions(torch.zeros((1,5,5)), 'ordinal').tolist() == [[200]*5]


def test_cache_matches_online_logits_gradients_and_order(tiny, tmp_path):
    from app.ml.features import cached_features
    _, tokenizer, model = tiny
    records = [{'id':'one','text':'a b c d e a b c d e'}, {'id':'two','text':'c a'}, {'id':'three','text':'e b d c'}]
    model.train(); model.encoder.eval()
    original = model(*encode_essays(tokenizer, [r['text'] for r in records], 8, 2), 3)
    scores = torch.tensor([[0,40,80,120,160],[200,160,120,80,40],[80,80,80,80,80]])
    model.loss(original,scores).backward()
    expected_grad = model.head.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    features = cached_features(model, tokenizer, records, 8, 2, tmp_path/'cache', batch_size=2)
    actual = model.forward_features(features)
    model.loss(actual,scores).backward()
    assert torch.allclose(original,actual,atol=1e-6)
    assert torch.allclose(model.head.weight.grad,expected_grad,atol=1e-6)
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.training and not model.encoder.training
    reversed_features = cached_features(model, tokenizer, records[::-1], 8, 2, tmp_path/'cache', batch_size=1)
    assert torch.allclose(features.flip(0),reversed_features,atol=1e-6)


def test_cache_invalidates_encoder_text_tokenizer_but_not_head_or_labels(tiny, tmp_path):
    from app.ml.features import cached_features
    _, tokenizer, model = tiny
    records = [{'id':'one','text':'a b c'}]; cache = tmp_path/'cache'
    original = cached_features(model,tokenizer,records,8,2,cache)
    assert len(list(cache.glob('*.pt'))) == 1
    with torch.no_grad(): model.head.weight.add_(1)
    records[0]['scores'] = [120]*5
    assert torch.equal(original, cached_features(model,tokenizer,records,8,2,cache))
    assert len(list(cache.glob('*.pt'))) == 1
    with torch.no_grad(): model.encoder.embeddings.word_embeddings.weight[5,0].add_(1)
    changed = cached_features(model,tokenizer,records,8,2,cache)
    assert len(list(cache.glob('*.pt'))) == 2 and not torch.equal(original,changed)
    records[0]['text'] = 'b c'
    cached_features(model,tokenizer,records,8,2,cache)
    assert len(list(cache.glob('*.pt'))) == 3
    tokenizer.add_tokens(['unused-new-token'])
    cached_features(model,tokenizer,records,8,2,cache)
    assert len(list(cache.glob('*.pt'))) == 4


def test_feature_cache_rejects_left_padding_and_non_fp32(tiny,tmp_path):
    from app.ml.features import cached_features
    _,tokenizer,model=tiny
    records=[{'id':'a','text':'a b'}]
    tokenizer.padding_side='left'
    with pytest.raises(ValueError,match='right padding'):
        cached_features(model,tokenizer,records,8,2,tmp_path/'cache')
    tokenizer.padding_side='right';model.double()
    with pytest.raises(ValueError,match='FP32'):
        cached_features(model,tokenizer,records,8,2,tmp_path/'cache')


def test_cache_rejects_trainable_encoder_and_detects_corruption(tiny,tmp_path):
    from app.ml.features import cached_features
    _,tokenizer,model=tiny
    records=[{'id':'one','text':'a b'}];cache=tmp_path/'cache'
    cached_features(model,tokenizer,records,8,2,cache)
    path=next(cache.glob('*.pt'))
    payload=torch.load(path,weights_only=True);payload['features'][0,0]+=1;torch.save(payload,path)
    with pytest.raises(ValueError,match='integrity'):
        cached_features(model,tokenizer,records,8,2,cache)
    model.encoder.embeddings.word_embeddings.weight.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):
        cached_features(model,tokenizer,records,8,2,cache)


def training_fixture(tmp_path):
    data=tmp_path/'data';data.mkdir()
    for split in ('training','development'):
        rows=[{'id':f'{split}:{i}','text':('a b c d e '* (i+1)),
               'scores':[40*((i+j)%6) for j in range(5)]} for i in range(5)]
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return data


def test_epoch_boundary_resume_does_not_repeat_completed_epoch(tiny,tmp_path,monkeypatch):
    import app.ml.train as train
    from app.ml.transformer import load_checkpoint
    base,_,_=tiny;data=training_fixture(tmp_path)
    options=dict(data_dir=data,base_model=base,epochs=2,batch_size=2,patience=5,max_length=8,
                 overlap=2,training_mode='frozen',learning_rate=.002)
    original_write=train.write_json
    def interrupt(path,payload):
        original_write(path,payload)
        if path.name=='training_log.json':
            raise InterruptedError('end of epoch')
    monkeypatch.setattr(train,'write_json',interrupt)
    with pytest.raises(InterruptedError): train.train_transformer(output=tmp_path/'resumed',**options)
    monkeypatch.setattr(train,'write_json',original_write)
    train.train_transformer(output=tmp_path/'resumed',resume=True,**options)
    train.train_transformer(output=tmp_path/'continuous',**options)
    a,_=load_checkpoint(tmp_path/'resumed'/'checkpoint-last');b,_=load_checkpoint(tmp_path/'continuous'/'checkpoint-last')
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    history=json.loads((tmp_path/'resumed'/'training_log.json').read_text())['epochs']
    assert [e['epoch'] for e in history]==[1,2]


def test_training_refuses_existing_experiment_before_loading_data(tmp_path):
    from app.ml.train import train_transformer
    output=tmp_path/'existing';output.mkdir();(output/'metadata.json').write_text('historic')
    with pytest.raises(FileExistsError):
        train_transformer(tmp_path/'missing',tmp_path/'missing-model',output)
    assert (output/'metadata.json').read_text()=='historic'


def test_external_resume_refuses_nonempty_destination_without_overwriting(tmp_path):
    from app.ml.train import train_transformer
    output=tmp_path/'existing';output.mkdir();(output/'metadata.json').write_text('historic')
    with pytest.raises(FileExistsError):
        train_transformer(tmp_path/'missing',tmp_path/'missing-model',output,
                          resume_from=tmp_path/'another-run'/'model')
    assert (output/'metadata.json').read_text()=='historic'


def test_completed_checkpoint_last_can_be_recovered_to_fresh_directory(tiny,tmp_path):
    from app.ml.train import train_transformer
    from app.ml.transformer import load_checkpoint
    base,_,_=tiny;data=training_fixture(tmp_path)
    options=dict(data_dir=data,base_model=base,epochs=1,batch_size=2,patience=2,max_length=8,
                 overlap=2,training_mode='frozen',learning_rate=.002)
    original=train_transformer(output=tmp_path/'source',**options)
    restored=train_transformer(output=tmp_path/'recovered',resume_from=tmp_path/'source'/'checkpoint-last',**options)
    assert restored['epochs_completed']==original['epochs_completed']==1
    a,ma=load_checkpoint(tmp_path/'source');b,mb=load_checkpoint(tmp_path/'recovered')
    assert ma['validation_metrics']==mb['validation_metrics']
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert (tmp_path/'recovered'/'training_log.json').is_file()
    assert (tmp_path/'recovered'/'checkpoint-last'/'trainer.pt').is_file()


def test_cached_training_matches_online_best_checkpoint(tiny,tmp_path):
    from app.ml.train import train_transformer
    from app.ml.transformer import load_checkpoint
    base,_,_=tiny;data=training_fixture(tmp_path)
    options=dict(data_dir=data,base_model=base,epochs=2,batch_size=2,patience=5,max_length=8,
                 overlap=2,training_mode='frozen',learning_rate=.002)
    train_transformer(output=tmp_path/'online',**options)
    train_transformer(output=tmp_path/'cached',embedding_cache=tmp_path/'features',**options)
    a,ma=load_checkpoint(tmp_path/'online');b,mb=load_checkpoint(tmp_path/'cached')
    for k,v in a.state_dict().items():
        assert torch.allclose(v,b.state_dict()[k],atol=1e-6), k
    assert ma['validation_metrics']==mb['validation_metrics']


def test_masked_tokens_do_not_change_cached_pooling_and_weights_include_specials(tiny):
    from types import SimpleNamespace
    _, _, model = tiny
    # Deterministic encoder exposes hand-computable CLS values: 1, 3, 7.
    class Encoder(torch.nn.Module):
        config = SimpleNamespace(hidden_size=1)
        def forward(self,input_ids,attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))
    scorer=EssayScorer(Encoder(),frozen=True)
    ids=torch.tensor([[1,2,0,0],[3,4,5,6],[7,8,0,0]])
    mask=(ids!=0).long();owners=torch.tensor([1,1,0])
    features=scorer.encode_features(ids,mask,owners,2)
    assert features[:,0].tolist()==pytest.approx([7.,7/3])
    # Real BERT must ignore IDs under mask=0 (right padding).
    model.eval()
    actual=model.encode_features(ids,mask,owners,2)
    ids[mask==0]=9
    assert torch.equal(actual,model.encode_features(ids,mask,owners,2))


@pytest.mark.parametrize('boundary', [False, True])
def test_periodic_resume_preserves_cursor_loss_and_best(tiny,tmp_path,monkeypatch,boundary):
    import app.ml.train as train
    from app.ml.checkpoints import BackupManager
    from app.ml.transformer import load_checkpoint
    base,_,_=tiny;data=training_fixture(tmp_path)
    options=dict(data_dir=data,base_model=base,epochs=2,batch_size=2,patience=5,max_length=8,
                 overlap=2,training_mode='frozen',learning_rate=.002,
                 backup_interval_seconds=1e-9)
    original=BackupManager.save
    saved=[]
    def interrupt(manager,writer):
        path=original(manager,writer)
        state=torch.load(path/'trainer_state.pt',weights_only=True)
        # Epoch one has completed: backups must carry its best weights.
        if state['epoch']==1 and (state.get('next_batch',0)==0 if boundary else state.get('next_batch',0)>0):
            saved.append(path)
            raise InterruptedError('backup persisted')
        return path
    monkeypatch.setattr(BackupManager,'save',interrupt)
    with pytest.raises(InterruptedError): train.train_transformer(output=tmp_path/'interrupted',**options)
    monkeypatch.setattr(BackupManager,'save',original)
    train.train_transformer(output=tmp_path/'restored',resume_from=saved[0]/'model',**options)
    train.train_transformer(output=tmp_path/'continuous',**options)
    a,ma=load_checkpoint(tmp_path/'restored');b,mb=load_checkpoint(tmp_path/'continuous')
    assert ma['validation_metrics']==mb['validation_metrics']
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    a,_=load_checkpoint(tmp_path/'restored'/'checkpoint-last');b,_=load_checkpoint(tmp_path/'continuous'/'checkpoint-last')
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    ha=json.loads((tmp_path/'restored'/'training_log.json').read_text())['epochs']
    hb=json.loads((tmp_path/'continuous'/'training_log.json').read_text())['epochs']
    assert [e['train_loss'] for e in ha]==[e['train_loss'] for e in hb]


def test_token_cache_identity_detects_replaced_vocab_at_same_path(tiny):
    from app.ml.transformer import token_cache_key
    _,tokenizer,_=tiny
    records=[{'id':'a','text':'a b'}]
    key=token_cache_key(tokenizer,records,8,2)
    tokenizer.add_tokens(['a-new-token'])
    assert key!=token_cache_key(tokenizer,records,8,2)
