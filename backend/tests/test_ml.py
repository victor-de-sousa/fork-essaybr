import json
import numpy as np
import pytest
from app.ml.predict import Predictor, ModelUnavailableError
from app.ml.baseline import train_baseline
from app.ml.transformer import chunk_token_ids, ordinal_targets, decode_predictions, EssayScorer, save_checkpoint, load_checkpoint


def test_missing_model_is_explicit(tmp_path):
    predictor=Predictor(tmp_path)
    assert predictor.status()['trained'] is False
    with pytest.raises(ModelUnavailableError): predictor.predict('Minha redação.')


def test_baseline_training_reload_and_validation_selection(tmp_path):
    data=tmp_path/'data';data.mkdir()
    for split,n in [('training',24),('development',6)]:
        with (data/f'{split}.jsonl').open('w') as f:
            for i in range(n):
                f.write(json.dumps({'id':f'{split}:{i}', 'text':f'educação escola sociedade futuro argumento {i}', 'scores':[40*(i%6)]*5})+'\n')
    # No testing file exists: training must only consume training/development.
    output=tmp_path/'model'
    report=train_baseline(data,output,alphas=[1,10],seed=42)
    prediction=Predictor(output).predict('educação escola sociedade')
    assert set(prediction['scores'])=={'c1','c2','c3','c4','c5'}
    assert all(v in range(0,201,40) for v in prediction['scores'].values())
    assert report['selection_split']=='development'
    assert len(report['candidates'])==2
    assert prediction['model_info']['trained'] is True


def test_chunking_does_not_drop_tail():
    ids=list(range(1050));chunks=chunk_token_ids(ids,510,64)
    assert set(x for chunk in chunks for x in chunk)==set(ids)
    assert chunks[-1][-1]==1049
    assert max(map(len,chunks))<=510
    with pytest.raises(ValueError): chunk_token_ids(ids,510,510)


def test_ordinal_targets_and_monotone_decoding():
    import torch
    scores=torch.tensor([[0,40,80,120,200]])
    assert ordinal_targets(scores).sum(-1).tolist()==[[0,1,2,3,5]]
    logits=torch.full((1,5,5), -10.);logits[:,:,0]=10
    assert decode_predictions(logits,'ordinal').tolist()==[[40]*5]


def test_tiny_transformer_real_gradient_checkpoint(tmp_path):
    import torch
    from transformers import BertConfig,BertModel
    torch.manual_seed(42)
    config=BertConfig(vocab_size=40,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32)
    model=EssayScorer(BertModel(config), 'ordinal')
    ids=torch.tensor([[1,2,3,0],[1,4,5,6],[1,7,0,0]])
    masks=(ids!=0).long();owners=torch.tensor([0,0,1]);scores=torch.tensor([[40,80,120,160,200],[0,40,80,120,160]])
    initial=model.head.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(), lr=.01)
    logits=model(ids,masks,owners,2)
    loss=model.loss(logits,scores);loss.backward();optimizer.step()
    assert not torch.equal(initial,model.head.weight)
    model.eval()
    with torch.no_grad(): expected=model(ids,masks,owners,2)
    save_checkpoint(model,tmp_path,{'trained':False,'engineering_smoke':True})
    restored,_=load_checkpoint(tmp_path)
    with torch.no_grad(): actual=restored(ids,masks,owners,2)
    assert torch.allclose(expected,actual,atol=1e-6)


def test_final_test_requires_explicit_gate_before_reading_any_data(tmp_path):
    from app.ml.evaluate import evaluate
    with pytest.raises(ValueError,match='final-test'):
        evaluate(tmp_path/'absent-data',tmp_path/'absent-model','testing',tmp_path/'out.json')


def test_ordinal_logits_are_monotone():
    import torch
    from transformers import BertConfig,BertModel
    config=BertConfig(vocab_size=40,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32)
    model=EssayScorer(BertModel(config),'ordinal').eval()
    logits=model(torch.tensor([[1,2,3]]),torch.ones((1,3),dtype=torch.long),torch.tensor([0]),1)
    assert torch.all(logits[:,:,1:] < logits[:,:,:-1])


def test_comparison_refuses_mixed_training_provenance(tmp_path):
    from app.ml.compare import compare
    for name,sha in [('first','aaa'),('second','bbb')]:
        output=tmp_path/name;output.mkdir()
        (output/'baseline.joblib').write_bytes(b'not loaded during comparison')
        (output/'metadata.json').write_text(json.dumps({'trained':True,'version':name,'kind':'tfidf_ridge',
            'selection_split':'development','validation_metrics':{'macro':{'qwk':.4,'mae':20}},
            'data_sha256':{'training':sha,'development':'ccc'}}))
    with pytest.raises(ValueError,match='provenance'):
        compare([tmp_path/'first',tmp_path/'second'],tmp_path/'selection.json')
