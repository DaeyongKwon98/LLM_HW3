#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from dpo_lora_train import load_jsonl, normalize_state_dict
from utils.FPT5ForConditionalGeneration import FPT5ForConditionalGeneration
from utils.model_utils import get_config, get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare base 3D-MolT5 checkpoints on caption-style generation samples.')
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--input_jsonl', default='data/dpo_chosen_rejected/test.jsonl')
    parser.add_argument('--output_dir', default='outputs/base_checkpoint_caption_compare')
    parser.add_argument('--num_per_task', type=int, default=6)
    parser.add_argument('--model_name', default='google/t5-v1_1-base')
    parser.add_argument('--molecule_dict', default='dict/selfies_dict.txt')
    parser.add_argument('--max_source_length', type=int, default=384)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--fp_bits', type=int, default=4096)
    parser.add_argument('--fp_level', type=int, default=3)
    parser.add_argument('--emb_setting', choices=['sum', 'concat'], default='sum')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--prompt_style', choices=['plain', 'instruction'], default='plain')
    return parser.parse_args()


def make_model_args(args: argparse.Namespace, checkpoint: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            name=args.model_name,
            checkpoint_path=checkpoint,
            dropout=0.0,
            random_init=False,
            compile=False,
        ),
        molecule_dict=args.molecule_dict,
        fp_bits=args.fp_bits,
        fp_level=args.fp_level,
        emb_setting=args.emb_setting,
        no_fp=False,
    )


def load_model(args: argparse.Namespace, checkpoint: str, tokenizer) -> FPT5ForConditionalGeneration:
    model_args = make_model_args(args, checkpoint)
    config = get_config(model_args)
    model = FPT5ForConditionalGeneration(config, model_args)
    model.resize_token_embeddings(len(tokenizer))
    state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(normalize_state_dict(state), strict=True)
    model.config.use_cache = False
    return model


def chosen_row(row: Dict) -> Dict:
    chosen = row.get('chosen') if isinstance(row.get('chosen'), dict) else {}
    out = dict(chosen)
    out.update({k: v for k, v in row.items() if k not in {'chosen', 'rejected'}})
    if 'e3fp' not in out and 'chosen_e3fp' in row:
        out['e3fp'] = row['chosen_e3fp']
    if 'caption_en' not in out and 'chosen_caption' in row:
        out['caption_en'] = row['chosen_caption']
    return out


def select_samples(rows: List[Dict], num_per_task: int) -> List[Dict]:
    selected = []
    for task in ['logp', 'qm9']:
        task_rows = [chosen_row(row) for row in rows if row.get('task') == task]
        if not task_rows:
            continue
        if task == 'logp':
            task_rows.sort(key=lambda row: float(row['logP']))
        else:
            task_rows.sort(key=lambda row: float(row['gap']))
        if len(task_rows) <= num_per_task:
            selected.extend(task_rows)
            continue
        indices = np.linspace(0, len(task_rows) - 1, num_per_task, dtype=int).tolist()
        selected.extend([task_rows[i] for i in indices])
    return selected


def build_plain_prompt(row: Dict) -> str:
    lines = [
        'Input:',
        f"SMILES: {row['smiles']}",
        f"SELFIES: <bom>{row['selfies']}<eom>",
    ]
    if row['task'] == 'logp':
        lines.append(f"LogP: {row['logP']}")
    elif row['task'] == 'qm9':
        lines.append(f"HOMO: {row['homo']}")
        lines.append(f"LUMO: {row['lumo']}")
        lines.append(f"GAP: {row['gap']}")
    else:
        raise ValueError(row['task'])
    lines.append('Caption:')
    return '\n'.join(lines)


def build_instruction_prompt(row: Dict) -> str:
    if row['task'] == 'logp':
        instruction = (
            'Task: Generate one molecular property caption. '
            'Use the exact LogP value, SMILES, SELFIES, and E3FP-conditioned molecular structure. '
            'Start with: LogP <value> indicates <lipophilicity category>. '
            'Then describe atom counts, ring systems, functional groups, HBD/HBA or TPSA when relevant, '
            'and explain the balance between nonpolar carbon/ring surface and polar contribution. '
            'Do not mention drug names, biological activity, targets, diseases, sources, or PubChem roles.'
        )
    elif row['task'] == 'qm9':
        instruction = (
            'Task: Generate one QM9 molecular property caption. '
            'Use the exact HOMO, LUMO, and HOMO-LUMO gap values with the SMILES, SELFIES, and E3FP-conditioned molecular structure. '
            'Start with: This QM9 molecule has HOMO <value>, LUMO <value>, and a HOMO-LUMO gap of <value> Hartree. '
            'Then describe atom counts, hetero atoms, rings, functional groups, and structural factors that rationalize the frontier orbital values. '
            'Do not mention drug names, biological activity, targets, diseases, sources, or PubChem roles.'
        )
    else:
        raise ValueError(row['task'])
    return instruction + '\n' + build_plain_prompt(row)


def build_prompt(row: Dict, prompt_style: str = 'plain') -> str:
    if prompt_style == 'plain':
        return build_plain_prompt(row)
    if prompt_style == 'instruction':
        return build_instruction_prompt(row)
    raise ValueError(prompt_style)

def normalize_fp(fp: List[List[int]], fp_dim: int) -> np.ndarray:
    fp_array = np.asarray(fp, dtype=np.int64)
    if fp_array.ndim == 1:
        fp_array = fp_array.reshape(-1, fp_dim)
    if fp_array.shape[1] < fp_dim:
        padded = np.full((fp_array.shape[0], fp_dim), -1, dtype=np.int64)
        padded[:, : fp_array.shape[1]] = fp_array
        fp_array = padded
    elif fp_array.shape[1] > fp_dim:
        fp_array = fp_array[:, :fp_dim]
    return fp_array


def fp_span(input_ids: np.ndarray, bom_id: int, eom_id: int, max_source_length: int) -> Tuple[int, int]:
    bom_positions = np.where(input_ids == bom_id)[0]
    if len(bom_positions) == 0:
        return 0, max_source_length
    start = int(bom_positions[0]) + 1
    eom_positions = np.where(input_ids[start:] == eom_id)[0]
    stop = start + int(eom_positions[0]) if len(eom_positions) > 0 else max_source_length
    return start, stop


def encode_batch(rows: List[Dict], tokenizer, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    prompts = [build_prompt(row, args.prompt_style) for row in rows]
    tokenized = tokenizer(
        prompts,
        return_attention_mask=True,
        return_tensors='np',
        padding='max_length',
        truncation=True,
        max_length=args.max_source_length,
    )
    input_ids = tokenized['input_ids'].astype(np.int64)
    molecule_fp_ids = np.full((len(rows), args.max_source_length, args.fp_level + 1), -1, dtype=np.int64)
    bom_id = tokenizer.convert_tokens_to_ids('<bom>')
    eom_id = tokenizer.convert_tokens_to_ids('<eom>')
    for i, row in enumerate(rows):
        fp_array = normalize_fp(row['e3fp'], args.fp_level + 1)
        start, stop = fp_span(input_ids[i], bom_id, eom_id, args.max_source_length)
        n = min(len(fp_array), max(0, stop - start))
        if n > 0:
            molecule_fp_ids[i, start:start+n, :] = fp_array[:n]
    return {
        'prompts': prompts,
        'input_ids': torch.from_numpy(input_ids),
        'attention_mask': torch.from_numpy(tokenized['attention_mask'].astype(np.int64)),
        'molecule_fp_ids': torch.from_numpy(molecule_fp_ids),
    }


def batches(rows: List[Dict], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def sanitize(path: str) -> str:
    return Path(path).stem.replace('.', '_').replace('-', '_')


def main() -> None:
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    rows = select_samples(load_jsonl(args.input_jsonl), args.num_per_task)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    samples_path = Path(args.output_dir) / 'selected_samples.jsonl'
    with samples_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps({
                'id': row.get('id'),
                'task': row['task'],
                'smiles': row['smiles'],
                'selfies': row['selfies'],
                'logP': row.get('logP'),
                'homo': row.get('homo'),
                'lumo': row.get('lumo'),
                'gap': row.get('gap'),
                'reference': row['caption_en'],
                'prompt_style': args.prompt_style,
            }, ensure_ascii=False) + '\n')

    combined_path = Path(args.output_dir) / 'all_predictions.jsonl'
    with combined_path.open('w', encoding='utf-8') as combined_f:
        for checkpoint in args.checkpoints:
            model_id = sanitize(checkpoint)
            print(f'Loading {checkpoint}')
            model_args = make_model_args(args, checkpoint)
            tokenizer = get_tokenizer(model_args)
            model = load_model(args, checkpoint, tokenizer).to(device)
            model.eval()
            per_model_path = Path(args.output_dir) / f'{model_id}_predictions.jsonl'
            with per_model_path.open('w', encoding='utf-8') as f:
                for batch_rows in batches(rows, args.batch_size):
                    encoded = encode_batch(batch_rows, tokenizer, args)
                    prompts = encoded.pop('prompts')
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    with torch.no_grad():
                        seq = model.generate(
                            input_ids=encoded['input_ids'],
                            attention_mask=encoded['attention_mask'],
                            molecule_fp_ids=encoded['molecule_fp_ids'],
                            max_new_tokens=args.max_new_tokens,
                            num_beams=args.num_beams,
                        )
                    preds = tokenizer.batch_decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for row, prompt, pred in zip(batch_rows, prompts, preds):
                        out = {
                            'model_id': model_id,
                            'checkpoint': checkpoint,
                            'id': row.get('id'),
                            'task': row['task'],
                            'smiles': row['smiles'],
                            'prompt_style': args.prompt_style,
                            'prompt': prompt,
                            'prediction': pred.strip(),
                            'reference': row['caption_en'],
                            'prediction_words': len(pred.strip().split()),
                            'reference_words': len(row['caption_en'].split()),
                        }
                        for key in ['logP', 'homo', 'lumo', 'gap']:
                            if key in row:
                                out[key] = row[key]
                        line = json.dumps(out, ensure_ascii=False)
                        f.write(line + '\n')
                        combined_f.write(line + '\n')
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f'Wrote {per_model_path}')
    print(f'Wrote {combined_path}')


if __name__ == '__main__':
    main()
