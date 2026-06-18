import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import get_linear_schedule_with_warmup

from utils.FPT5ForConditionalGeneration import FPT5ForConditionalGeneration
from utils.model_utils import get_config, get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DPO LoRA training for 3D-MolT5 preference pairs. Positive/negative "
            "pairs share SELFIES/properties/caption and differ in E3FP; the loss "
            "prefers the caption under the positive E3FP condition."
        )
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/t5-v1_1-base")
    parser.add_argument("--molecule_dict", type=str, default="dict/selfies_dict.txt")

    parser.add_argument("--logp_positive", type=str, default="training_data/logp_positive.jsonl")
    parser.add_argument("--logp_negative", type=str, default="training_data/logp_final_negative_pair.jsonl")
    parser.add_argument("--qm9_positive", type=str, default="training_data/qm9_positive.jsonl")
    parser.add_argument("--qm9_negative", type=str, default="training_data/qm9_final_negative_pair.jsonl")
    parser.add_argument(
        "--caption_pair_file",
        type=str,
        default="",
        help="Optional JSONL with DPO fields. If train/eval_pair_file are unset, this file is split by eval_ratio.",
    )
    parser.add_argument(
        "--train_pair_file",
        type=str,
        default="",
        help="Explicit train JSONL with DPO fields. Supports chosen_e3fp/rejected_e3fp or shared e3fp caption pairs.",
    )
    parser.add_argument(
        "--eval_pair_file",
        type=str,
        default="",
        help="Explicit eval JSONL with DPO fields. Supports chosen_e3fp/rejected_e3fp or shared e3fp caption pairs.",
    )
    parser.add_argument(
        "--positive_file",
        type=str,
        default="",
        help="Optional aligned positive JSONL for single-task mDPO/CoPO training.",
    )
    parser.add_argument(
        "--negative_file",
        type=str,
        default="",
        help="Optional aligned negative JSONL for single-task mDPO/CoPO training.",
    )
    parser.add_argument(
        "--property_task",
        choices=["auto", "logp", "qm9", "mixed"],
        default="auto",
        help="Task type for mDPO data. Use mixed/auto for files containing task/property_task per row.",
    )
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument("--max_source_length", type=int, default=320)
    parser.add_argument("--max_target_length", type=int, default=768)
    parser.add_argument("--fp_bits", type=int, default=4096)
    parser.add_argument("--fp_level", type=int, default=3)
    parser.add_argument("--emb_setting", choices=["sum", "concat"], default="sum")
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--tuning_mode",
        choices=["lora", "full"],
        default="lora",
        help="Train LoRA adapters only, or update all policy-model parameters.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,wi,wi_0,wi_1,wo",
        help="Comma-separated final module names of nn.Linear layers to LoRA-wrap.",
    )
    parser.add_argument("--train_fp_embedding", action="store_true")

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--preference_loss",
        choices=["dpo", "mdpo", "copo"],
        default="dpo",
        help="dpo compares chosen/rejected responses; mdpo/copo compares condition-level likelihoods with a shared caption.",
    )
    parser.add_argument(
        "--logprob_reduction",
        choices=["sum", "mean"],
        default="sum",
        help="Sequence log-prob reduction used in the DPO ratio.",
    )
    parser.add_argument("--sft_loss_weight", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=0.0, help="Anchor margin for mDPO/CoPO.")
    parser.add_argument("--lambda_sft", type=float, default=1.0)
    parser.add_argument("--lambda_copo", type=float, default=0.2)
    parser.add_argument("--lambda_anchor", type=float, default=0.05)
    parser.add_argument(
        "--prompt_style",
        choices=["plain", "instruction", "mdpo_instruction"],
        default="plain",
        help="Prompt template for source construction.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument(
        "--best_model_metric",
        type=str,
        default="eval_loss",
        help="Evaluation metric used to select output_dir/best_model.",
    )
    parser.add_argument(
        "--greater_is_better",
        action="store_true",
        help="Use this when best_model_metric should be maximized instead of minimized.",
    )
    parser.add_argument("--save_merged", action="store_true", default=True)
    parser.add_argument("--no_save_merged", dest="save_merged", action="store_false")
    parser.add_argument("--strict_pair_check", action="store_true", default=True)
    parser.add_argument("--no_strict_pair_check", dest="strict_pair_check", action="store_false")
    parser.add_argument("--dry_run_data", action="store_true")
    parser.add_argument("--smoke_test", action="store_true", help="Run one forward/backward step and exit.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            dropout=args.dropout,
            random_init=False,
            compile=False,
        ),
        molecule_dict=args.molecule_dict,
        fp_bits=args.fp_bits,
        fp_level=args.fp_level,
        emb_setting=args.emb_setting,
        no_fp=False,
    )


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_base_model(args: argparse.Namespace, tokenizer) -> FPT5ForConditionalGeneration:
    model_args = make_model_args(args)
    config = get_config(model_args)
    model = FPT5ForConditionalGeneration(config, model_args)
    model.resize_token_embeddings(len(tokenizer))
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = normalize_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.config.use_cache = False
    return model


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[torch.nn.Parameter]:
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + delta

    def merged_linear(self) -> nn.Linear:
        merged = copy.deepcopy(self.base_layer)
        delta = torch.matmul(self.lora_B.weight, self.lora_A.weight) * self.scaling
        merged.weight.data.add_(delta.to(dtype=merged.weight.dtype, device=merged.weight.device))
        return merged


def inject_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    r: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    target_set = {name.strip() for name in target_modules if name.strip()}
    matched: List[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module_name.split(".")[-1] in target_set:
            matched.append(module_name)

    for module_name in matched:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        child = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return matched


def merge_lora_inplace(model: nn.Module) -> None:
    for module_name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, child_name = module_name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, module.merged_linear())


def trainable_parameter_report(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def enable_input_grads_for_checkpointing(model: FPT5ForConditionalGeneration) -> None:
    def make_output_require_grad(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)

    seen = set()
    for embedding in [model.get_input_embeddings(), model.encoder.embed_tokens, model.decoder.embed_tokens]:
        if embedding is not None and id(embedding) not in seen:
            embedding.register_forward_hook(make_output_require_grad)
            seen.add(id(embedding))


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def canonical_task(row: Dict, fallback: str = "") -> str:
    task = row.get("property_task") or row.get("task") or fallback
    task = str(task).lower()
    if task == "logp":
        return "logp"
    if task == "qm9":
        return "qm9"
    raise ValueError(f"Unsupported task: {task}")


def first_present(row: Dict, keys: Sequence[str]):
    for key in keys:
        if key in row:
            return row[key]
    return None


def canonical_property_row(row: Dict, task: str) -> Dict:
    out = dict(row)
    if task == "logp":
        value = first_present(out, ["logP", "logp", "LogP"])
        if value is None:
            raise ValueError("Missing LogP/logP field")
        out["logP"] = value
    elif task == "qm9":
        homo = first_present(out, ["homo", "HOMO"])
        lumo = first_present(out, ["lumo", "LUMO"])
        gap = first_present(out, ["gap", "homo_lumo_gap", "HOMO_LUMO_gap", "gap_homo_lumo"])
        if homo is None or lumo is None or gap is None:
            raise ValueError("Missing HOMO/LUMO/gap fields")
        out["homo"] = homo
        out["lumo"] = lumo
        out["gap"] = gap
    else:
        raise ValueError(f"Unsupported task: {task}")
    return out


def same_property_fields(task: str) -> Tuple[str, ...]:
    if task == "logp":
        return ("smiles", "selfies", "logP", "caption_en")
    if task == "qm9":
        return ("smiles", "selfies", "homo", "lumo", "gap", "caption_en")
    raise ValueError(f"Unsupported task: {task}")


def build_source(row: Dict, task: str, prompt_style: str = "plain") -> str:
    row = canonical_property_row(row, task)
    lines: List[str] = []
    if prompt_style in {"instruction", "mdpo_instruction"}:
        if task == "logp":
            lines.extend([
                "Instruction: Generate one molecular property caption.",
                "Use the 1D SELFIES representation, the supplied 3D E3FP condition, and the exact LogP value.",
                "Explain atom counts, ring systems, functional groups, and the balance between nonpolar surface and polar contribution.",
                "Do not describe biological activity, drug roles, targets, diseases, or PubChem metadata.",
            ])
        elif task == "qm9":
            lines.extend([
                "Instruction: Generate one QM9 electronic-property caption.",
                "Use the 1D SELFIES representation, the supplied 3D E3FP condition, and the exact HOMO, LUMO, and HOMO-LUMO gap values.",
                "Explain structural factors that rationalize the frontier orbital values and gap.",
                "Do not describe biological activity, drug roles, targets, diseases, or PubChem metadata.",
            ])
    lines.append("Input:")
    if prompt_style in {"instruction", "mdpo_instruction"} and row.get("smiles"):
        lines.append(f"SMILES: {row['smiles']}")
    lines.append(f"SELFIES: <bom>{row['selfies']}<eom>")
    if task == "logp":
        lines.append(f"LogP: {row['logP']}")
        if prompt_style == "mdpo_instruction":
            lines.append("Question: Explain the LogP-related molecular property of this molecule using both the 1D and 3D representations.")
    elif task == "qm9":
        lines.append(f"HOMO: {row['homo']}")
        lines.append(f"LUMO: {row['lumo']}")
        lines.append(f"GAP: {row['gap']}")
        if prompt_style == "mdpo_instruction":
            lines.append("Question: Explain the electronic molecular property of this molecule using both the 1D and 3D representations.")
    else:
        raise ValueError(f"Unsupported task: {task}")
    lines.append("Caption:")
    return "\n".join(lines)


@dataclass
class PreferenceExample:
    task: str
    chosen_source: str
    rejected_source: str
    chosen_fp: List[List[int]]
    rejected_fp: List[List[int]]
    chosen_response: str
    rejected_response: str


class E3FPPairDPODataset(Dataset):
    def __init__(
        self,
        pair_specs: Sequence[Tuple[str, str, str]],
        strict_pair_check: bool = True,
        max_samples: int = 0,
        prompt_style: str = "plain",
    ) -> None:
        self.examples: List[PreferenceExample] = []
        for task, positive_path, negative_path in pair_specs:
            positives = load_jsonl(positive_path)
            negatives = load_jsonl(negative_path)
            if len(positives) != len(negatives):
                raise ValueError(
                    f"{task}: positive/negative lengths differ: "
                    f"{len(positives)} vs {len(negatives)}"
                )
            fields = same_property_fields(task)
            for idx, (positive, negative) in enumerate(zip(positives, negatives)):
                positive = canonical_property_row(positive, task)
                negative = canonical_property_row(negative, task)
                if strict_pair_check:
                    for field in fields:
                        if positive.get(field) != negative.get(field):
                            raise ValueError(
                                f"{task} row {idx} differs in {field}; this trainer "
                                "expects negative pairs to differ only in E3FP."
                            )
                self.examples.append(
                    PreferenceExample(
                        task=task,
                        chosen_source=build_source(positive, task, prompt_style),
                        rejected_source=build_source(negative, task, prompt_style),
                        chosen_fp=positive["e3fp"],
                        rejected_fp=negative["e3fp"],
                        chosen_response=positive["caption_en"].strip(),
                        rejected_response=negative.get("caption_en", positive["caption_en"]).strip(),
                    )
                )
        if max_samples > 0:
            self.examples = self.examples[:max_samples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreferenceExample:
        return self.examples[index]



def nested_value(row: Dict, side: str, key: str):
    nested = row.get(side)
    if isinstance(nested, dict):
        return nested.get(key)
    return None


def dpo_caption(row: Dict, side: str) -> Optional[str]:
    for key in (f"{side}_caption", "caption_en", "caption"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = nested_value(row, side, "caption_en") or nested_value(row, side, "caption")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def dpo_e3fp(row: Dict, side: str):
    side_key = f"{side}_e3fp"
    if side_key in row:
        return row[side_key]
    value = nested_value(row, side, "e3fp")
    if value is not None:
        return value
    if "e3fp" in row:
        return row["e3fp"]
    raise KeyError(f"Missing {side} E3FP")


class CaptionPairDPODataset(Dataset):
    def __init__(self, path: str, max_samples: int = 0, prompt_style: str = "plain") -> None:
        self.examples: List[PreferenceExample] = []
        rows = load_jsonl(path)
        for idx, row in enumerate(rows):
            task = canonical_task(row)
            row = canonical_property_row(row, task)
            if task not in {"logp", "qm9"}:
                raise ValueError(f"Unsupported task in {path} row {idx}: {task}")
            chosen_caption = dpo_caption(row, "chosen")
            rejected_caption = dpo_caption(row, "rejected")
            if not chosen_caption or not rejected_caption:
                raise ValueError(f"Missing chosen/rejected caption in {path} row {idx}")
            try:
                chosen_fp = dpo_e3fp(row, "chosen")
                rejected_fp = dpo_e3fp(row, "rejected")
            except KeyError as exc:
                raise ValueError(f"{exc} in {path} row {idx}") from exc
            self.examples.append(
                PreferenceExample(
                    task=task,
                    chosen_source=build_source(row, task, prompt_style),
                    rejected_source=build_source(row, task, prompt_style),
                    chosen_fp=chosen_fp,
                    rejected_fp=rejected_fp,
                    chosen_response=chosen_caption,
                    rejected_response=rejected_caption,
                )
            )
        if max_samples > 0:
            self.examples = self.examples[:max_samples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreferenceExample:
        return self.examples[index]


def caption_from_row(row: Dict) -> Optional[str]:
    for key in ["caption_en", "chosen_caption", "caption", "output"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = row.get("chosen")
    if isinstance(nested, dict):
        for key in ["caption_en", "caption"]:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def condition_e3fp(row: Dict, side: str):
    if side == "pos":
        keys = ["e3fp_pos", "positive_e3fp", "chosen_e3fp", "e3fp"]
        nested_names = ["positive", "chosen"]
    else:
        keys = ["e3fp_neg", "negative_e3fp", "rejected_e3fp", "original_rejected_e3fp"]
        nested_names = ["negative", "rejected"]
    for key in keys:
        if key in row:
            return row[key]
    for nested_name in nested_names:
        nested = row.get(nested_name)
        if isinstance(nested, dict) and "e3fp" in nested:
            return nested["e3fp"]
    raise KeyError(f"Missing {side} E3FP")


class MDPOConditionDataset(Dataset):
    def __init__(
        self,
        path: str = "",
        pair_specs: Optional[Sequence[Tuple[str, str, str]]] = None,
        strict_pair_check: bool = True,
        max_samples: int = 0,
        prompt_style: str = "mdpo_instruction",
        property_task: str = "auto",
    ) -> None:
        self.examples: List[PreferenceExample] = []
        if path:
            self._load_single_file(path, max_samples=max_samples, prompt_style=prompt_style, property_task=property_task)
        elif pair_specs:
            self._load_aligned_files(
                pair_specs=pair_specs,
                strict_pair_check=strict_pair_check,
                max_samples=max_samples,
                prompt_style=prompt_style,
                property_task=property_task,
            )
        else:
            raise ValueError("MDPOConditionDataset requires path or pair_specs")

    def _fallback_task(self, row: Dict, property_task: str, fallback: str = "") -> str:
        if property_task in {"logp", "qm9"}:
            return property_task
        return canonical_task(row, fallback=fallback)

    def _load_single_file(self, path: str, max_samples: int, prompt_style: str, property_task: str) -> None:
        rows = load_jsonl(path)
        for idx, row in enumerate(rows):
            task = self._fallback_task(row, property_task)
            row = canonical_property_row(row, task)
            caption = caption_from_row(row)
            if not caption:
                raise ValueError(f"Missing caption in {path} row {idx}")
            try:
                pos_fp = condition_e3fp(row, "pos")
                neg_fp = condition_e3fp(row, "neg")
            except KeyError as exc:
                raise ValueError(f"{exc} in {path} row {idx}") from exc
            source = build_source(row, task, prompt_style)
            self.examples.append(
                PreferenceExample(
                    task=task,
                    chosen_source=source,
                    rejected_source=source,
                    chosen_fp=pos_fp,
                    rejected_fp=neg_fp,
                    chosen_response=caption,
                    rejected_response=caption,
                )
            )
        if max_samples > 0:
            self.examples = self.examples[:max_samples]

    def _load_aligned_files(
        self,
        pair_specs: Sequence[Tuple[str, str, str]],
        strict_pair_check: bool,
        max_samples: int,
        prompt_style: str,
        property_task: str,
    ) -> None:
        for fallback_task, positive_path, negative_path in pair_specs:
            positives = load_jsonl(positive_path)
            negatives = load_jsonl(negative_path)
            if len(positives) != len(negatives):
                raise ValueError(
                    f"{fallback_task}: positive/negative lengths differ: {len(positives)} vs {len(negatives)}"
                )
            for idx, (positive, negative) in enumerate(zip(positives, negatives)):
                task = self._fallback_task(positive, property_task, fallback=fallback_task)
                positive = canonical_property_row(positive, task)
                negative = canonical_property_row(negative, task)
                caption = caption_from_row(positive)
                if not caption:
                    raise ValueError(f"Missing caption in {positive_path} row {idx}")
                if strict_pair_check:
                    for field in same_property_fields(task):
                        if positive.get(field) != negative.get(field):
                            raise ValueError(
                                f"{task} row {idx} differs in {field}; mDPO expects only E3FP to differ."
                            )
                    if positive.get("e3fp") == negative.get("e3fp"):
                        raise ValueError(f"{task} row {idx} has identical positive/negative E3FP")
                source_pos = build_source(positive, task, prompt_style)
                source_neg = build_source(negative, task, prompt_style)
                self.examples.append(
                    PreferenceExample(
                        task=task,
                        chosen_source=source_pos,
                        rejected_source=source_neg,
                        chosen_fp=positive["e3fp"],
                        rejected_fp=negative["e3fp"],
                        chosen_response=caption,
                        rejected_response=caption,
                    )
                )
        if max_samples > 0:
            self.examples = self.examples[:max_samples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreferenceExample:
        return self.examples[index]


TASK_TO_ID = {"logp": 0, "qm9": 1}
ID_TO_TASK = {value: key for key, value in TASK_TO_ID.items()}


class DPODataCollator:
    def __init__(
        self,
        tokenizer,
        max_source_length: int,
        max_target_length: int,
        fp_level: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.fp_dim = fp_level + 1
        self.pad_token_id = tokenizer.pad_token_id
        self.bom_id = tokenizer.convert_tokens_to_ids("<bom>")
        self.eom_id = tokenizer.convert_tokens_to_ids("<eom>")

    def __call__(self, examples: Sequence[PreferenceExample]) -> Dict[str, torch.Tensor]:
        chosen = self._encode_side(
            [example.chosen_source for example in examples],
            [example.chosen_response for example in examples],
            [example.chosen_fp for example in examples],
        )
        rejected = self._encode_side(
            [example.rejected_source for example in examples],
            [example.rejected_response for example in examples],
            [example.rejected_fp for example in examples],
        )
        batch: Dict[str, torch.Tensor] = {}
        for key, value in chosen.items():
            batch[f"chosen_{key}"] = value
        for key, value in rejected.items():
            batch[f"rejected_{key}"] = value
        batch["task_ids"] = torch.tensor([TASK_TO_ID[example.task] for example in examples], dtype=torch.long)
        return batch

    def _encode_side(
        self,
        sources: Sequence[str],
        responses: Sequence[str],
        fps: Sequence[List[List[int]]],
    ) -> Dict[str, torch.Tensor]:
        source_tokens = self.tokenizer(
            list(sources),
            return_attention_mask=True,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_source_length,
        )
        target_tokens = self.tokenizer(
            list(responses),
            return_attention_mask=False,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
        )

        labels = target_tokens["input_ids"].astype(np.int64)
        labels[labels == self.pad_token_id] = -100
        molecule_fp_ids = np.full(
            (len(sources), self.max_source_length, self.fp_dim),
            -1,
            dtype=np.int64,
        )
        input_ids = source_tokens["input_ids"].astype(np.int64)
        for row_idx, fp in enumerate(fps):
            fp_array = self._normalize_fp(fp)
            if fp_array.shape[0] == 0:
                continue
            fp_start, fp_stop = self._fp_span(input_ids[row_idx])
            if fp_start >= fp_stop:
                continue
            n = min(fp_array.shape[0], fp_stop - fp_start)
            molecule_fp_ids[row_idx, fp_start : fp_start + n, :] = fp_array[:n]

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(source_tokens["attention_mask"].astype(np.int64)),
            "labels": torch.from_numpy(labels),
            "molecule_fp_ids": torch.from_numpy(molecule_fp_ids),
        }

    def _normalize_fp(self, fp: List[List[int]]) -> np.ndarray:
        fp_array = np.asarray(fp, dtype=np.int64)
        if fp_array.ndim == 1:
            fp_array = fp_array.reshape(-1, self.fp_dim)
        if fp_array.shape[1] < self.fp_dim:
            padded = np.full((fp_array.shape[0], self.fp_dim), -1, dtype=np.int64)
            padded[:, : fp_array.shape[1]] = fp_array
            fp_array = padded
        elif fp_array.shape[1] > self.fp_dim:
            fp_array = fp_array[:, : self.fp_dim]
        return fp_array

    def _fp_span(self, input_ids: np.ndarray) -> Tuple[int, int]:
        bom_positions = np.where(input_ids == self.bom_id)[0]
        if len(bom_positions) == 0:
            return 0, self.max_source_length
        fp_start = int(bom_positions[0]) + 1
        eom_positions = np.where(input_ids[fp_start:] == self.eom_id)[0]
        if len(eom_positions) > 0:
            fp_stop = fp_start + int(eom_positions[0])
        else:
            fp_stop = self.max_source_length
        return fp_start, fp_stop


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def sequence_logprob(
    model: FPT5ForConditionalGeneration,
    batch: Dict[str, torch.Tensor],
    prefix: str,
    reduction: str = "sum",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = batch[f"{prefix}_labels"]
    outputs = model(
        input_ids=batch[f"{prefix}_input_ids"],
        attention_mask=batch[f"{prefix}_attention_mask"],
        molecule_fp_ids=batch[f"{prefix}_molecule_fp_ids"],
        labels=labels,
        use_cache=False,
    )
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    valid_mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid_mask, 0)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * valid_mask
    seq_logprob_sum = token_log_probs.sum(dim=-1)
    n_tokens = valid_mask.sum(dim=-1).clamp_min(1)
    sft_nll = -seq_logprob_sum / n_tokens
    if reduction == "mean":
        seq_logprob = seq_logprob_sum / n_tokens
    else:
        seq_logprob = seq_logprob_sum
    return seq_logprob, sft_nll, n_tokens


def sequence_log_probs(
    model: FPT5ForConditionalGeneration,
    batch: Dict[str, torch.Tensor],
    prefix: str,
    reduction: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_logprob, _sft_nll, lengths = sequence_logprob(model, batch, prefix, reduction)
    return seq_logprob, lengths


def detached_mean(tensor: torch.Tensor) -> float:
    return float(tensor.detach().mean().cpu())


def add_task_metrics(
    metrics: Dict[str, float],
    task_ids: Optional[torch.Tensor],
    per_example: Dict[str, torch.Tensor],
) -> None:
    if task_ids is None:
        return
    for task_id, task_name in ID_TO_TASK.items():
        mask = task_ids.eq(task_id)
        if not bool(mask.any()):
            continue
        for name, values in per_example.items():
            metrics[f"{name}_{task_name}"] = detached_mean(values[mask])


def mdpo_loss(
    policy_model: FPT5ForConditionalGeneration,
    reference_model: FPT5ForConditionalGeneration,
    batch: Dict[str, torch.Tensor],
    beta: float,
    reduction: str,
    delta: float,
    lambda_sft: float,
    lambda_copo: float,
    lambda_anchor: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not torch.equal(batch["chosen_labels"], batch["rejected_labels"]):
        raise ValueError("mDPO/CoPO requires identical positive and negative labels")

    labels = batch["chosen_labels"]
    if not bool(labels.eq(-100).any()):
        raise ValueError("Expected -100 padding positions in labels")

    fp_diff = batch["chosen_molecule_fp_ids"].ne(batch["rejected_molecule_fp_ids"]).flatten(1).any(dim=1)
    if not bool(fp_diff.any()):
        raise ValueError("mDPO/CoPO requires at least one positive/negative E3FP difference in the batch")

    logp_pos, sft_nll_pos, _ = sequence_logprob(policy_model, batch, "chosen", reduction)
    logp_neg, _sft_nll_neg, _ = sequence_logprob(policy_model, batch, "rejected", reduction)
    with torch.no_grad():
        ref_logp_pos, _ref_sft_pos, _ = sequence_logprob(reference_model, batch, "chosen", reduction)
        ref_logp_neg, _ref_sft_neg, _ = sequence_logprob(reference_model, batch, "rejected", reduction)

    reward_pos = beta * (logp_pos - ref_logp_pos)
    reward_neg = beta * (logp_neg - ref_logp_neg)
    reward_margin = reward_pos - reward_neg

    loss_sft_per = sft_nll_pos
    loss_copo_per = -F.logsigmoid(reward_margin)
    loss_anchor_per = -F.logsigmoid(reward_pos - delta)

    loss_sft = loss_sft_per.mean()
    loss_copo = loss_copo_per.mean()
    loss_anchor = loss_anchor_per.mean()
    loss = lambda_sft * loss_sft + lambda_copo * loss_copo
    if lambda_anchor > 0:
        loss = loss + lambda_anchor * loss_anchor

    if not torch.isfinite(loss):
        raise FloatingPointError("mDPO/CoPO loss became non-finite")

    metrics = {
        "loss": detached_mean(loss),
        "loss_sft": detached_mean(loss_sft),
        "loss_copo": detached_mean(loss_copo),
        "loss_anchor": detached_mean(loss_anchor),
        "logp_pos_mean": detached_mean(logp_pos),
        "logp_neg_mean": detached_mean(logp_neg),
        "logp_margin_mean": detached_mean(logp_pos - logp_neg),
        "ref_logp_pos_mean": detached_mean(ref_logp_pos),
        "ref_logp_neg_mean": detached_mean(ref_logp_neg),
        "reward_pos_mean": detached_mean(reward_pos),
        "reward_neg_mean": detached_mean(reward_neg),
        "reward_margin_mean": detached_mean(reward_margin),
        "accuracy": detached_mean((reward_pos > reward_neg).float()),
        "condition_fp_diff_rate": detached_mean(fp_diff.float()),
    }
    add_task_metrics(
        metrics,
        batch.get("task_ids"),
        {
            "loss": lambda_sft * loss_sft_per + lambda_copo * loss_copo_per + lambda_anchor * loss_anchor_per,
            "loss_sft": loss_sft_per,
            "loss_copo": loss_copo_per,
            "loss_anchor": loss_anchor_per,
            "reward_margin": reward_margin,
            "logp_pos": logp_pos,
            "logp_neg": logp_neg,
        },
    )
    return loss, metrics


def dpo_loss(
    policy_model: FPT5ForConditionalGeneration,
    reference_model: FPT5ForConditionalGeneration,
    batch: Dict[str, torch.Tensor],
    beta: float,
    reduction: str,
    sft_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    policy_chosen_logps, chosen_lengths = sequence_log_probs(policy_model, batch, "chosen", reduction)
    policy_rejected_logps, _ = sequence_log_probs(policy_model, batch, "rejected", reduction)
    with torch.no_grad():
        ref_chosen_logps, _ = sequence_log_probs(reference_model, batch, "chosen", reduction)
        ref_rejected_logps, _ = sequence_log_probs(reference_model, batch, "rejected", reduction)

    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - reference_logratios)
    losses = -F.logsigmoid(logits)
    loss = losses.mean()

    if sft_loss_weight > 0:
        # Keep this normalized by target length so the auxiliary term is stable
        # when switching between sum and mean DPO reductions.
        chosen_nll = -(policy_chosen_logps / chosen_lengths if reduction == "sum" else policy_chosen_logps)
        loss = loss + sft_loss_weight * chosen_nll.mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "reward_chosen": float(chosen_rewards.mean().cpu()),
        "reward_rejected": float(rejected_rewards.mean().cpu()),
        "reward_margin": float((chosen_rewards - rejected_rewards).mean().cpu()),
        "accuracy": float((chosen_rewards > rejected_rewards).float().mean().cpu()),
        "policy_logratio": float(policy_logratios.detach().mean().cpu()),
        "reference_logratio": float(reference_logratios.detach().mean().cpu()),
    }
    return loss, metrics


def compute_preference_loss(
    policy_model: FPT5ForConditionalGeneration,
    reference_model: FPT5ForConditionalGeneration,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if args.preference_loss in {"mdpo", "copo"}:
        return mdpo_loss(
            policy_model,
            reference_model,
            batch,
            beta=args.beta,
            reduction=args.logprob_reduction,
            delta=args.delta,
            lambda_sft=args.lambda_sft,
            lambda_copo=args.lambda_copo,
            lambda_anchor=args.lambda_anchor,
        )
    return dpo_loss(
        policy_model,
        reference_model,
        batch,
        beta=args.beta,
        reduction=args.logprob_reduction,
        sft_loss_weight=args.sft_loss_weight,
    )


@torch.no_grad()
def evaluate(
    policy_model: FPT5ForConditionalGeneration,
    reference_model: FPT5ForConditionalGeneration,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    amp_dtype: Optional[torch.dtype],
    max_batches: int = 20,
) -> Dict[str, float]:
    policy_model.eval()
    reference_model.eval()
    sums: Dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            _, metrics = compute_preference_loss(
                policy_model,
                reference_model,
                batch,
                args,
            )
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        count += 1
    policy_model.train()
    if count == 0:
        return {}
    return {f"eval_{key}": value / count for key, value in sums.items()}


def adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for name, tensor in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            state[name] = tensor.detach().cpu()
        elif name.startswith("encoder.molecule_fp_embed_tokens") and tensor.requires_grad:
            state[name] = tensor.detach().cpu()
    return state


def save_adapter(
    model: nn.Module,
    tokenizer,
    args: argparse.Namespace,
    output_dir: str,
    matched_modules: Sequence[str],
    extra_metadata: Optional[Dict] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(adapter_state_dict(model), os.path.join(output_dir, "adapter_model.bin"))
    with open(os.path.join(output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        metadata = {
            "base_model_name": args.model_name,
            "base_checkpoint_path": args.checkpoint_path,
            "caption_pair_file": args.caption_pair_file,
            "train_pair_file": args.train_pair_file,
            "eval_pair_file": args.eval_pair_file,
            "positive_file": args.positive_file,
            "negative_file": args.negative_file,
            "property_task": args.property_task,
            "preference_loss": args.preference_loss,
            "beta": args.beta,
            "delta": args.delta,
            "lambda_sft": args.lambda_sft,
            "lambda_copo": args.lambda_copo,
            "lambda_anchor": args.lambda_anchor,
            "prompt_style": args.prompt_style,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": list(args.lora_target_modules.split(",")),
            "matched_modules": list(matched_modules),
            "fp_bits": args.fp_bits,
            "fp_level": args.fp_level,
            "emb_setting": args.emb_setting,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        json.dump(metadata, f, indent=2)
    tokenizer.save_pretrained(output_dir)


def save_full_model(
    model: nn.Module,
    tokenizer,
    args: argparse.Namespace,
    output_dir: str,
    extra_metadata: Optional[Dict] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model_merged.bin"))
    with open(os.path.join(output_dir, "full_finetune_config.json"), "w", encoding="utf-8") as f:
        metadata = {
            "base_model_name": args.model_name,
            "base_checkpoint_path": args.checkpoint_path,
            "caption_pair_file": args.caption_pair_file,
            "train_pair_file": args.train_pair_file,
            "eval_pair_file": args.eval_pair_file,
            "positive_file": args.positive_file,
            "negative_file": args.negative_file,
            "property_task": args.property_task,
            "preference_loss": args.preference_loss,
            "beta": args.beta,
            "delta": args.delta,
            "lambda_sft": args.lambda_sft,
            "lambda_copo": args.lambda_copo,
            "lambda_anchor": args.lambda_anchor,
            "prompt_style": args.prompt_style,
            "tuning_mode": args.tuning_mode,
            "fp_bits": args.fp_bits,
            "fp_level": args.fp_level,
            "emb_setting": args.emb_setting,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        json.dump(metadata, f, indent=2)
    tokenizer.save_pretrained(output_dir)


def save_policy_checkpoint(
    model: nn.Module,
    tokenizer,
    args: argparse.Namespace,
    output_dir: str,
    matched_modules: Sequence[str],
    extra_metadata: Optional[Dict] = None,
) -> None:
    if args.tuning_mode == "full":
        save_full_model(model, tokenizer, args, output_dir, extra_metadata=extra_metadata)
    else:
        save_adapter(model, tokenizer, args, output_dir, matched_modules, extra_metadata=extra_metadata)


def save_final(
    model: nn.Module,
    tokenizer,
    args: argparse.Namespace,
    matched_modules: Sequence[str],
) -> None:
    if args.tuning_mode == "full":
        save_full_model(model, tokenizer, args, args.output_dir)
    else:
        save_adapter(model, tokenizer, args, args.output_dir, matched_modules)
    if args.tuning_mode == "lora" and args.save_merged:
        merge_lora_inplace(model)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model_merged.bin"))


def format_metrics(metrics: Dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


def run_smoke_test(
    policy_model: FPT5ForConditionalGeneration,
    reference_model: FPT5ForConditionalGeneration,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    amp_dtype: Optional[torch.dtype],
) -> None:
    for parameter in reference_model.parameters():
        if parameter.requires_grad:
            raise AssertionError("Reference model has trainable parameters")
    batch = next(iter(dataloader))
    batch = move_batch(batch, device)
    policy_model.train()
    reference_model.eval()
    policy_model.zero_grad(set_to_none=True)
    reference_model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None and device.type == "cuda",
    ):
        loss, metrics = compute_preference_loss(policy_model, reference_model, batch, args)
    if not torch.isfinite(loss):
        raise AssertionError("Smoke-test loss is not finite")
    loss.backward()
    policy_has_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        for parameter in policy_model.parameters()
        if parameter.requires_grad
    )
    if not policy_has_grad:
        raise AssertionError("No finite nonzero gradients found on policy trainable parameters")
    ref_has_grad = any(parameter.grad is not None for parameter in reference_model.parameters())
    if ref_has_grad:
        raise AssertionError("Reference model received gradients")
    policy_model.zero_grad(set_to_none=True)
    reference_model.zero_grad(set_to_none=True)
    print(f"Smoke test passed: {format_metrics(metrics)}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    if args.mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    model_args = make_model_args(args)
    tokenizer = get_tokenizer(model_args)
    explicit_pair_split = bool(args.train_pair_file)
    use_mdpo = args.preference_loss in {"mdpo", "copo"}
    if use_mdpo:
        mdpo_prompt_style = args.prompt_style
        if explicit_pair_split:
            train_dataset = MDPOConditionDataset(
                path=args.train_pair_file,
                max_samples=args.max_samples,
                prompt_style=mdpo_prompt_style,
                property_task=args.property_task,
            )
            eval_dataset = (
                MDPOConditionDataset(
                    path=args.eval_pair_file,
                    prompt_style=mdpo_prompt_style,
                    property_task=args.property_task,
                )
                if args.eval_pair_file
                else []
            )
            dataset = train_dataset
        elif args.positive_file and args.negative_file:
            fallback_task = args.property_task if args.property_task in {"logp", "qm9"} else ""
            dataset = MDPOConditionDataset(
                pair_specs=[(fallback_task, args.positive_file, args.negative_file)],
                strict_pair_check=args.strict_pair_check,
                max_samples=args.max_samples,
                prompt_style=mdpo_prompt_style,
                property_task=args.property_task,
            )
        else:
            dataset = MDPOConditionDataset(
                pair_specs=[
                    ("logp", args.logp_positive, args.logp_negative),
                    ("qm9", args.qm9_positive, args.qm9_negative),
                ],
                strict_pair_check=args.strict_pair_check,
                max_samples=args.max_samples,
                prompt_style=mdpo_prompt_style,
                property_task=args.property_task,
            )
    elif explicit_pair_split:
        train_dataset = CaptionPairDPODataset(
            path=args.train_pair_file,
            max_samples=args.max_samples,
            prompt_style=args.prompt_style,
        )
        eval_dataset = CaptionPairDPODataset(path=args.eval_pair_file, prompt_style=args.prompt_style) if args.eval_pair_file else []
        dataset = train_dataset
    elif args.caption_pair_file:
        dataset = CaptionPairDPODataset(
            path=args.caption_pair_file,
            max_samples=args.max_samples,
            prompt_style=args.prompt_style,
        )
    else:
        dataset = E3FPPairDPODataset(
            pair_specs=[
                ("logp", args.logp_positive, args.logp_negative),
                ("qm9", args.qm9_positive, args.qm9_negative),
            ],
            strict_pair_check=args.strict_pair_check,
            max_samples=args.max_samples,
            prompt_style=args.prompt_style,
        )
    collator = DPODataCollator(
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        fp_level=args.fp_level,
    )

    if len(dataset) == 0:
        raise ValueError("No DPO examples loaded")
    if not explicit_pair_split:
        eval_size = int(len(dataset) * args.eval_ratio)
        if args.eval_ratio > 0 and eval_size == 0 and len(dataset) > 1:
            eval_size = 1
        train_size = len(dataset) - eval_size
        train_dataset, eval_dataset = random_split(
            dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    print(f"Loaded {args.preference_loss} pairs: train={len(train_dataset)} eval={len(eval_dataset)}")
    if args.dry_run_data:
        sample_batch = collator([dataset[0]])
        for key, value in sample_batch.items():
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    policy_model = load_base_model(args, tokenizer)
    reference_model = load_base_model(args, tokenizer)
    matched_modules: List[str] = []
    if args.tuning_mode == "lora":
        for parameter in policy_model.parameters():
            parameter.requires_grad = False
        matched_modules = inject_lora(
            policy_model,
            target_modules=args.lora_target_modules.split(","),
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        if not matched_modules:
            raise ValueError(f"No LoRA target modules matched: {args.lora_target_modules}")
        if args.train_fp_embedding:
            for parameter in policy_model.encoder.molecule_fp_embed_tokens.parameters():
                parameter.requires_grad = True
    else:
        for parameter in policy_model.parameters():
            parameter.requires_grad = True

    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        enable_input_grads_for_checkpointing(policy_model)

    policy_model.to(device)
    reference_model.to(device)
    reference_model.eval()

    trainable, total = trainable_parameter_report(policy_model)
    if args.tuning_mode == "lora":
        print(f"LoRA-wrapped modules: {len(matched_modules)}")
    else:
        print("Tuning mode: full fine-tuning")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100.0 * trainable / total:.4f}%)")

    if args.smoke_test:
        run_smoke_test(policy_model, reference_model, train_loader, device, args, amp_dtype)
        return

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16" and device.type == "cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running: Dict[str, float] = {}
    running_count = 0
    best_metric_value: Optional[float] = None
    best_model_dir = os.path.join(args.output_dir, "best_model")

    policy_model.train()
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            micro_step += 1
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                loss, metrics = compute_preference_loss(
                    policy_model,
                    reference_model,
                    batch,
                    args,
                )
                scaled_loss = loss / args.grad_accum_steps

            scaler.scale(scaled_loss).backward()
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value
            running_count += 1

            should_step = (
                micro_step % args.grad_accum_steps == 0
                or batch_idx == len(train_loader) - 1
            )
            if not should_step:
                continue

            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy_model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0 or global_step == 1:
                averaged = {key: value / running_count for key, value in running.items()}
                averaged["lr"] = scheduler.get_last_lr()[0]
                print(f"step={global_step} epoch={epoch + 1} {format_metrics(averaged)}")
                running.clear()
                running_count = 0

            if len(eval_dataset) > 0 and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                eval_metrics = evaluate(
                    policy_model,
                    reference_model,
                    eval_loader,
                    device,
                    args,
                    amp_dtype,
                )
                print(f"step={global_step} {format_metrics(eval_metrics)}")
                if args.best_model_metric in eval_metrics:
                    metric_value = eval_metrics[args.best_model_metric]
                    is_best = (
                        best_metric_value is None
                        or (args.greater_is_better and metric_value > best_metric_value)
                        or (not args.greater_is_better and metric_value < best_metric_value)
                    )
                    if is_best:
                        best_metric_value = metric_value
                        save_policy_checkpoint(
                            policy_model,
                            tokenizer,
                            args,
                            best_model_dir,
                            matched_modules,
                            extra_metadata={
                                "best_model_metric": args.best_model_metric,
                                "best_metric_value": best_metric_value,
                                "best_global_step": global_step,
                                "best_eval_metrics": eval_metrics,
                            },
                        )
                        with open(os.path.join(best_model_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "best_model_metric": args.best_model_metric,
                                    "best_metric_value": best_metric_value,
                                    "best_global_step": global_step,
                                    "best_eval_metrics": eval_metrics,
                                },
                                f,
                                indent=2,
                            )
                        print(
                            f"Saved best model to {best_model_dir} "
                            f"({args.best_model_metric}={best_metric_value:.6f}, step={global_step})"
                        )
                else:
                    print(
                        f"Best-model metric '{args.best_model_metric}' not found in eval metrics; "
                        f"available={sorted(eval_metrics.keys())}"
                    )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                step_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_policy_checkpoint(policy_model, tokenizer, args, step_dir, matched_modules)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if len(eval_dataset) > 0:
        eval_metrics = evaluate(policy_model, reference_model, eval_loader, device, args, amp_dtype)
        print(f"final {format_metrics(eval_metrics)}")
        if args.best_model_metric in eval_metrics:
            metric_value = eval_metrics[args.best_model_metric]
            is_best = (
                best_metric_value is None
                or (args.greater_is_better and metric_value > best_metric_value)
                or (not args.greater_is_better and metric_value < best_metric_value)
            )
            if is_best:
                best_metric_value = metric_value
                save_policy_checkpoint(
                    policy_model,
                    tokenizer,
                    args,
                    best_model_dir,
                    matched_modules,
                    extra_metadata={
                        "best_model_metric": args.best_model_metric,
                        "best_metric_value": best_metric_value,
                        "best_global_step": global_step,
                        "best_eval_metrics": eval_metrics,
                    },
                )
                with open(os.path.join(best_model_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "best_model_metric": args.best_model_metric,
                            "best_metric_value": best_metric_value,
                            "best_global_step": global_step,
                            "best_eval_metrics": eval_metrics,
                        },
                        f,
                        indent=2,
                    )
                print(
                    f"Saved best model to {best_model_dir} "
                    f"({args.best_model_metric}={best_metric_value:.6f}, step={global_step})"
                )

    save_final(policy_model, tokenizer, args, matched_modules)
    if args.tuning_mode == "full":
        print(f"Saved full fine-tuned model to {args.output_dir}/pytorch_model_merged.bin")
    else:
        print(f"Saved LoRA adapter to {args.output_dir}/adapter_model.bin")
        if args.save_merged:
            print(f"Saved merged model to {args.output_dir}/pytorch_model_merged.bin")


if __name__ == "__main__":
    main()
