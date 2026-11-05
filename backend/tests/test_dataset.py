import csv
from pathlib import Path
import pytest
from app.ml.dataset import parse_row, prepare_dataset, load_records


def row(text="['Primeiro parágrafo.', 'Segundo parágrafo.']", scores='[40, 80, 120, 160, 200]', total='600'):
    return {'essay':text, 'competence':scores, 'score':total, 'prompt':'1', 'title':'Título'}


def write_split(root, name, rows):
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'{name}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(row()));writer.writeheader();writer.writerows(rows)


def test_paragraphs_and_targets_are_preserved():
    record=parse_row(row(), 'training', 2)
    assert record['text']=='Primeiro parágrafo.\n\nSegundo parágrafo.'
    assert record['scores']==[40,80,120,160,200]
    assert record['total']==600


@pytest.mark.parametrize('updates', [
    {'competence':'[40,80,120,160,201]'}, {'competence':'[40]'},
    {'score':'599'}, {'competence':'[True,80,120,160,200]'},
    {'essay':"__import__('pathlib').Path('/tmp/ml-malicious').touch()"}, {'essay':"['']"},
])
def test_invalid_rows_are_rejected(updates):
    with pytest.raises(ValueError): parse_row(row(**{}) | updates, 'training', 2)


def test_cross_split_duplicates_quarantined_without_modifying_sources(tmp_path):
    source=tmp_path/'sources';out=tmp_path/'prepared'
    long=' '.join(f'palavra{i}' for i in range(100))
    write_split(source,'training',[row(repr([long])), row(repr(['texto exclusivo treino']))])
    write_split(source,'development',[row(repr([long+' adicional']))])
    write_split(source,'testing',[row(repr([long]))])
    before={p.name:p.read_bytes() for p in source.glob('*.csv')}
    report=prepare_dataset(source,out,near_threshold=.85)
    assert len(load_records(out/'training.jsonl'))==1
    assert load_records(out/'development.jsonl')==[]
    assert len(load_records(out/'testing.jsonl'))==1
    assert report['quarantined_count']==2
    assert report['duplicate_groups'][0]['cross_split'] is True
    assert {p.name:p.read_bytes() for p in source.glob('*.csv')}==before
    assert (out/'manifest.json').exists()


def test_prepared_integrity_rejects_modified_split(tmp_path):
    from app.ml.dataset import verify_prepared
    source=tmp_path/'source';prepared=tmp_path/'prepared'
    for name in ('training','development','testing'):
        write_split(source,name,[row(repr([name+' texto único']))])
    prepare_dataset(source,prepared)
    (prepared/'training.jsonl').write_text('tampered\n')
    with pytest.raises(ValueError,match='checksum'):
        verify_prepared(prepared,('training','development'))
