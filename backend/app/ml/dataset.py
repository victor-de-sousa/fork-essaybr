"""Validate the original Essay-BR splits and quarantine leakage without editing sources."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

SPLITS = ('training', 'development', 'testing')
ALLOWED_SCORES = (0, 40, 80, 120, 160, 200)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_row(row: dict, split: str, line: int) -> dict:
    try:
        paragraphs = ast.literal_eval(row['essay'])
        scores = ast.literal_eval(row['competence'])
        total = int(row['score'])
    except (ValueError, SyntaxError, TypeError, KeyError) as exc:
        raise ValueError(f'Invalid serialized row: {exc}') from exc
    if not isinstance(paragraphs, list) or not paragraphs or any(not isinstance(p, str) for p in paragraphs):
        raise ValueError('essay must be a nonempty list of paragraph strings')
    if not any(p.strip() for p in paragraphs):
        raise ValueError('Essay is empty')
    if not isinstance(scores, list) or len(scores) != 5 or any(type(s) is not int or s not in ALLOWED_SCORES for s in scores):
        raise ValueError('competence requires exactly five integers in 0..200, step 40')
    if sum(scores) != total:
        raise ValueError('Total score differs from competency sum')
    text = '\n\n'.join(paragraphs)
    return {'id': f'{split}:{line}', 'split': split, 'source_line': line,
            'prompt': row.get('prompt', ''), 'title': row.get('title', ''),
            'paragraphs': paragraphs, 'text': text, 'scores': scores, 'total': total,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest()}


def normalize(text: str) -> str:
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def shingles(text: str) -> set[str]:
    words = normalize(text).split()
    return {' '.join(words[i:i + 3]) for i in range(max(1, len(words) - 2))}


def duplicate_clusters(records: list[dict], threshold: float) -> list[list[int]]:
    """Exact Jaccard verification with lossless global-prefix candidate filtering.

    Threshold is fixed a priori, not selected against test labels. Only text is
    inspected to prevent cross-split leakage; test targets never guide training.
    """
    if not 0 < threshold <= 1:
        raise ValueError('near_threshold must be in (0, 1]')
    sets = [shingles(record['text']) for record in records]
    frequencies = Counter(token for tokens in sets for token in tokens)
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    postings = defaultdict(list)
    for i, tokens in enumerate(sets):
        ordered = sorted(tokens, key=lambda s: (frequencies[s], s))
        prefix = ordered[:len(tokens) - math.ceil(threshold * len(tokens)) + 1]
        candidates = {other for token in prefix for other in postings[token]}
        for j in candidates:
            other = sets[j]
            if min(len(tokens), len(other)) < threshold * max(len(tokens), len(other)):
                continue
            similarity = len(tokens & other) / len(tokens | other)
            if similarity >= threshold:
                parent[find(i)] = find(j)
        for token in prefix:
            postings[token].append(i)
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    return [group for group in groups.values() if len(group) > 1]


def load_records(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def verify_prepared(directory: Path, splits=('training', 'development')) -> None:
    """Verify only requested splits; training never opens testing targets."""
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if not manifest_path.exists():
        return  # Programmatic fixtures may supply already validated JSONL directly.
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for split in splits:
        expected = manifest['files'][split]['sha256']
        if file_hash(directory / f'{split}.jsonl') != expected:
            raise ValueError(f'Prepared dataset checksum mismatch: {split}')


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path: Path, records) -> None:
    with path.open('w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def distribution(records):
    return {'count': len(records), 'competencies': {
        f'c{i + 1}': dict(sorted(Counter(str(r['scores'][i]) for r in records).items())) for i in range(5)},
        'total': dict(sorted(Counter(str(r['total']) for r in records).items())),
        'prompts': dict(sorted(Counter(r['prompt'] for r in records).items())),
        'words': {'min': min((len(r['text'].split()) for r in records), default=0),
                  'max': max((len(r['text'].split()) for r in records), default=0)}}


def prepare_dataset(source: Path, output: Path, near_threshold: float = .85) -> dict:
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('Prepared output must be outside the original corpus directory')
    records, invalid, sources, original_counts = [], [], {}, {}
    for split in SPLITS:
        path = source / f'{split}.csv'
        sources[split] = {'path': str(path.resolve()), 'sha256': file_hash(path)}
        with path.open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        original_counts[split] = len(rows)
        for line, row in enumerate(rows, 2):
            try:
                records.append(parse_row(row, split, line))
            except ValueError as exc:
                invalid.append({'id': f'{split}:{line}', 'reason': str(exc)})
    groups = duplicate_clusters(records, near_threshold)
    removed, group_reports = set(), []
    priority = {'testing': 0, 'development': 1, 'training': 2}
    for group in groups:
        ordered = sorted(group, key=lambda i: (priority[records[i]['split']], records[i]['source_line']))
        # Preserve the held-out test population, including any within-test duplicates.
        keep = [i for i in ordered if records[i]['split'] == 'testing'] or ordered[:1]
        quarantine = set(ordered) - set(keep)
        removed.update(quarantine)
        group_reports.append({'members': [records[i]['id'] for i in ordered],
            'retained': [records[i]['id'] for i in keep],
            'cross_split': len({records[i]['split'] for i in group}) > 1,
            'kind': 'exact' if len({normalize(records[i]['text']) for i in group}) == 1 else 'near'})
    output.mkdir(parents=True, exist_ok=True)
    clean = {split: [r for i, r in enumerate(records) if r['split'] == split and i not in removed] for split in SPLITS}
    for split, rows in clean.items():
        write_jsonl(output / f'{split}.jsonl', rows)
    quarantined = [records[i] | {'reason': 'exact_or_near_duplicate_cluster'} for i in sorted(removed)]
    write_jsonl(output / 'quarantine.jsonl', quarantined)
    report = {'schema_version': 1, 'near_duplicate_metric': 'word_trigram_jaccard',
        'near_threshold': near_threshold, 'priority': 'testing > development > training',
        'test_labels_used_for_selection': False, 'original_counts': original_counts,
        'invalid_rows': invalid, 'invalid_count': len(invalid),
        'quarantined_count': len(removed), 'duplicate_groups': group_reports,
        'splits': {split: distribution(rows) for split, rows in clean.items()}}
    write_json(output / 'audit.json', report)
    manifest = {'schema_version': 1, 'sources': sources,
        'files': {split: {'name': f'{split}.jsonl', 'sha256': file_hash(output / f'{split}.jsonl'), 'count': len(clean[split])} for split in SPLITS},
        'audit_sha256': file_hash(output / 'audit.json')}
    write_json(output / 'manifest.json', manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('essay-br/splits'))
    parser.add_argument('--output', type=Path, default=Path('datasets/prepared'))
    parser.add_argument('--near-threshold', type=float, default=.85)
    args = parser.parse_args()
    report = prepare_dataset(args.source, args.output, args.near_threshold)
    print(json.dumps({'original': report['original_counts'], 'retained': {s: d['count'] for s, d in report['splits'].items()},
                      'invalid': report['invalid_count'], 'quarantined': report['quarantined_count']}, indent=2))

if __name__ == '__main__':
    main()
