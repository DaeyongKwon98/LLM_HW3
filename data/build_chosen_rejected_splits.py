#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def assert_pair_alignment(task: str, pos: Dict, neg: Dict, idx: int) -> None:
    fields = ['smiles', 'selfies', 'caption_en']
    if task == 'logp':
        fields.append('logP')
    elif task == 'qm9':
        fields.extend(['homo', 'lumo', 'gap'])
    else:
        raise ValueError(task)
    for field in fields:
        if pos.get(field) != neg.get(field):
            raise ValueError(f'{task} row {idx} mismatch in {field}')


def make_pair(task: str, idx: int, pos: Dict, neg: Dict) -> Dict:
    assert_pair_alignment(task, pos, neg, idx)
    pair = {
        'id': f'{task}_{idx:06d}',
        'task': task,
        'smiles': pos['smiles'],
        'selfies': pos['selfies'],
        'chosen': pos,
        'rejected': neg,
        # Convenience aliases for loaders that expect separated fields.
        'chosen_caption': pos['caption_en'],
        'rejected_caption': neg['caption_en'],
        'chosen_e3fp': pos['e3fp'],
        'rejected_e3fp': neg['e3fp'],
    }
    if task == 'logp':
        pair['logP'] = pos['logP']
    else:
        pair['homo'] = pos['homo']
        pair['lumo'] = pos['lumo']
        pair['gap'] = pos['gap']
    return pair


def build_pairs(task: str, positive_path: Path, negative_path: Path) -> List[Dict]:
    positives = load_jsonl(positive_path)
    negatives = load_jsonl(negative_path)
    if len(positives) != len(negatives):
        raise ValueError(f'{task}: positive/negative length mismatch {len(positives)} vs {len(negatives)}')
    return [make_pair(task, idx, pos, neg) for idx, (pos, neg) in enumerate(zip(positives, negatives))]


def split_811(rows: List[Dict], seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    return rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:]


def summarize(split_rows: Dict[str, List[Dict]]) -> Dict:
    stats = {}
    for split, rows in split_rows.items():
        by_task = Counter(row['task'] for row in rows)
        caption_words = [len(row['chosen_caption'].split()) for row in rows]
        chosen_fp_lens = [len(row['chosen_e3fp']) for row in rows]
        rejected_fp_lens = [len(row['rejected_e3fp']) for row in rows]
        same_caption = sum(row['chosen_caption'] == row['rejected_caption'] for row in rows)
        same_e3fp = sum(row['chosen_e3fp'] == row['rejected_e3fp'] for row in rows)
        stats[split] = {
            'num_pairs': len(rows),
            'by_task': dict(by_task),
            'same_caption_pairs': same_caption,
            'same_e3fp_pairs': same_e3fp,
            'chosen_caption_words': {
                'min': min(caption_words) if caption_words else 0,
                'mean': round(mean(caption_words), 3) if caption_words else 0,
                'max': max(caption_words) if caption_words else 0,
            },
            'chosen_e3fp_len': {
                'min': min(chosen_fp_lens) if chosen_fp_lens else 0,
                'mean': round(mean(chosen_fp_lens), 3) if chosen_fp_lens else 0,
                'max': max(chosen_fp_lens) if chosen_fp_lens else 0,
            },
            'rejected_e3fp_len': {
                'min': min(rejected_fp_lens) if rejected_fp_lens else 0,
                'mean': round(mean(rejected_fp_lens), 3) if rejected_fp_lens else 0,
                'max': max(rejected_fp_lens) if rejected_fp_lens else 0,
            },
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('data'))
    parser.add_argument('--output_dir', type=Path, default=Path('data/dpo_chosen_rejected'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    specs = [
        ('logp', args.data_dir / 'logp_positive.jsonl', args.data_dir / 'logp_negative.jsonl'),
        ('qm9', args.data_dir / 'qm9_positive.jsonl', args.data_dir / 'qm9_negative.jsonl'),
    ]
    split_rows = {'train': [], 'val': [], 'test': []}
    task_split_counts = {}
    for task, positive_path, negative_path in specs:
        pairs = build_pairs(task, positive_path, negative_path)
        train, val, test = split_811(pairs, seed=args.seed)
        split_rows['train'].extend(train)
        split_rows['val'].extend(val)
        split_rows['test'].extend(test)
        task_split_counts[task] = {'train': len(train), 'val': len(val), 'test': len(test), 'total': len(pairs)}

    # Shuffle combined task mixture in each split, deterministically.
    for i, split in enumerate(['train', 'val', 'test']):
        rng = random.Random(args.seed + 1000 + i)
        rng.shuffle(split_rows[split])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        write_jsonl(args.output_dir / f'{split}.jsonl', rows)

    stats = summarize(split_rows)
    stats['seed'] = args.seed
    stats['task_split_counts'] = task_split_counts
    stats['schema'] = {
        'chosen': 'full positive row, including correct E3FP and caption_en',
        'rejected': 'full negative row, same molecule/properties/caption_en but negative E3FP',
        'convenience_fields': ['chosen_caption', 'rejected_caption', 'chosen_e3fp', 'rejected_e3fp'],
    }
    (args.output_dir / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
