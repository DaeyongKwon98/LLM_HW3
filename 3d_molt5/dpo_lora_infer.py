#!/usr/bin/env python3
import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dpo_lora_train import (
    LoRALinear,
    build_source,
    inject_lora,
    load_jsonl,
    merge_lora_inplace,
    normalize_state_dict,
)
from utils.FPT5ForConditionalGeneration import FPT5ForConditionalGeneration
from utils.model_utils import get_config, get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for DPO LoRA-tuned 3D-MolT5 caption models.")
    parser.add_argument("--model_path", type=str, required=True, help="Directory or file for merged model or LoRA adapter.")
    parser.add_argument("--base_checkpoint_path", type=str, default="model_checkpoint/3d-molt5-base-pubchem-cap.bin")
    parser.add_argument("--model_name", type=str, default="google/t5-v1_1-base")
    parser.add_argument("--molecule_dict", type=str, default="dict/selfies_dict.txt")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument(
        "--preference_side",
        choices=["chosen", "rejected", "negative_1d"],
        default="chosen",
        help="For pair JSONL, select which nested condition to use.",
    )
    parser.add_argument("--reference_field", type=str, default="auto", help="Reference caption field to copy into outputs. auto uses chosen_caption/caption_en/caption.")
    parser.add_argument("--prompt_style", choices=["plain", "instruction", "mdpo_instruction"], default="plain")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_source_length", type=int, default=320)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--fp_bits", type=int, default=4096)
    parser.add_argument("--fp_level", type=int, default=3)
    parser.add_argument("--emb_setting", choices=["sum", "concat"], default="sum")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,wi,wi_0,wi_1,wo")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return parser.parse_args()


def make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            name=args.model_name,
            checkpoint_path=args.base_checkpoint_path,
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


def resolve_model_files(model_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if os.path.isfile(model_path):
        if model_path.endswith(".bin"):
            return model_path, None, None
        raise ValueError(f"Unsupported model file: {model_path}")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(model_path)
    merged = os.path.join(model_path, "pytorch_model_merged.bin")
    adapter = os.path.join(model_path, "adapter_model.bin")
    adapter_config = os.path.join(model_path, "adapter_config.json")
    return (
        merged if os.path.exists(merged) else None,
        adapter if os.path.exists(adapter) else None,
        adapter_config if os.path.exists(adapter_config) else None,
    )


def load_base(args: argparse.Namespace, tokenizer) -> FPT5ForConditionalGeneration:
    model_args = make_model_args(args)
    config = get_config(model_args)
    model = FPT5ForConditionalGeneration(config, model_args)
    model.resize_token_embeddings(len(tokenizer))
    state = torch.load(args.base_checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(normalize_state_dict(state), strict=True)
    model.config.use_cache = False
    return model


def load_model(args: argparse.Namespace, tokenizer) -> FPT5ForConditionalGeneration:
    merged_file, adapter_file, adapter_config_file = resolve_model_files(args.model_path)
    model = load_base(args, tokenizer)

    if merged_file:
        state = torch.load(merged_file, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(normalize_state_dict(state), strict=True)
        return model

    if not adapter_file:
        raise FileNotFoundError(f"No pytorch_model_merged.bin or adapter_model.bin found under {args.model_path}")

    adapter_config = {}
    if adapter_config_file and os.path.exists(adapter_config_file):
        with open(adapter_config_file, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        args.lora_r = int(adapter_config.get("lora_r", args.lora_r))
        args.lora_alpha = float(adapter_config.get("lora_alpha", args.lora_alpha))
        args.lora_dropout = 0.0
        args.lora_target_modules = ",".join(adapter_config.get("lora_target_modules", args.lora_target_modules.split(",")))

    matched = inject_lora(
        model,
        target_modules=args.lora_target_modules.split(","),
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if not matched:
        raise ValueError("No LoRA modules matched while loading adapter")
    adapter_state = torch.load(adapter_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    unexpected = [key for key in unexpected if key]
    if unexpected:
        raise ValueError(f"Unexpected adapter keys: {unexpected[:10]}")
    merge_lora_inplace(model)
    return model


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


def get_nested_side(row: Dict, side: str) -> Dict:
    nested = row.get(side)
    return nested if isinstance(nested, dict) else {}


def get_row_value(row: Dict, key: str, side: Optional[str] = None):
    if side:
        nested = get_nested_side(row, side)
        if key in nested:
            return nested[key]
    if key in row:
        return row[key]
    return None


def get_e3fp(row: Dict, side: str):
    side_key = f"{side}_e3fp"
    if side_key in row:
        return row[side_key]
    if "e3fp" in row:
        return row["e3fp"]
    nested = get_nested_side(row, side)
    if "e3fp" in nested:
        return nested["e3fp"]
    raise KeyError(f"No E3FP found for side '{side}' in row {row.get('id', '<no id>')}")


def get_reference(row: Dict, args: argparse.Namespace) -> Optional[str]:
    if args.reference_field != "auto":
        return get_row_value(row, args.reference_field, args.preference_side)
    for key in [f"{args.preference_side}_caption", "caption_en", "caption", "output"]:
        value = get_row_value(row, key, args.preference_side)
        if value is not None:
            return value
    return None


def make_prompt_row(row: Dict, side: str) -> Dict:
    nested = get_nested_side(row, side)
    prompt_row = {k: v for k, v in row.items() if k not in {"chosen", "rejected"}}
    prompt_row.update(nested)
    return prompt_row


def encode_batch(rows: List[Dict], tokenizer, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    prompt_rows = [make_prompt_row(row, args.preference_side) for row in rows]
    prompts = [build_source(row, row["task"], args.prompt_style) for row in prompt_rows]
    tokenized = tokenizer(
        prompts,
        return_attention_mask=True,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=args.max_source_length,
    )
    input_ids = tokenized["input_ids"].astype(np.int64)
    molecule_fp_ids = np.full(
        (len(rows), args.max_source_length, args.fp_level + 1),
        -1,
        dtype=np.int64,
    )
    bom_id = tokenizer.convert_tokens_to_ids("<bom>")
    eom_id = tokenizer.convert_tokens_to_ids("<eom>")
    for i, row in enumerate(rows):
        fp_array = normalize_fp(get_e3fp(row, args.preference_side), args.fp_level + 1)
        start, stop = fp_span(input_ids[i], bom_id, eom_id, args.max_source_length)
        n = min(len(fp_array), max(0, stop - start))
        if n > 0:
            molecule_fp_ids[i, start : start + n, :] = fp_array[:n]
    return {
        "prompts": prompts,
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(tokenized["attention_mask"].astype(np.int64)),
        "molecule_fp_ids": torch.from_numpy(molecule_fp_ids),
    }


def decode_predictions(tokenizer, sequences: torch.Tensor) -> List[str]:
    # The 3D-MolT5 tokenizer marks molecule tokens as special tokens, and in this
    # tokenizer setup skip_special_tokens=True also drops numeric pieces during
    # T5 decoding. Decode raw text and remove only control/sentinel tokens so
    # numeric property values such as QM9, LogP, HOMO/LUMO, atom counts, and TPSA
    # are preserved.
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    control_tokens = [tokenizer.pad_token, tokenizer.eos_token, tokenizer.unk_token, "<bom>", "<eom>"]
    cleaned = []
    for text in decoded:
        for token in control_tokens:
            if token:
                text = text.replace(token, "")
        # T5 sentinel tokens should not appear in captions; remove them if generated.
        import re
        text = re.sub(r"<extra_id_\d+>", "", text)
        text = " ".join(text.split())
        text = re.sub(r"\b3\s+D\b", "3D", text)
        text = re.sub(r"(?<=\d)\s*/\s*(?=\d)", "/", text)
        text = re.sub(r"\bA2\b", "A^2", text)
        cleaned.append(text)
    return cleaned


def batched(rows: List[Dict], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    model_args = make_model_args(args)
    tokenizer = get_tokenizer(model_args)
    model = load_model(args, tokenizer)
    model.eval()
    model.to(device)
    if args.dtype == "float16":
        model.to(torch.float16)
    elif args.dtype == "bfloat16":
        model.to(torch.bfloat16)

    rows = load_jsonl(args.input_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for batch_rows in batched(rows, args.batch_size):
            encoded = encode_batch(batch_rows, tokenizer, args)
            prompts = encoded.pop("prompts")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                sequences = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    molecule_fp_ids=encoded["molecule_fp_ids"],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            predictions = decode_predictions(tokenizer, sequences)
            for row, prompt, prediction in zip(batch_rows, prompts, predictions):
                out = {
                    "id": row.get("id"),
                    "task": row["task"],
                    "smiles": get_row_value(row, "smiles", args.preference_side),
                    "selfies": get_row_value(row, "selfies", args.preference_side),
                    "preference_side": args.preference_side,
                    "prompt_style": args.prompt_style,
                    "prompt": prompt,
                    "prediction": prediction,
                    "reference": get_reference(row, args),
                }
                for key in ["logP", "homo", "lumo", "gap", "chosen_caption", "rejected_caption"]:
                    if key in row:
                        out[key] = row[key]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
