#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated captions against references.")
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--prediction_field", default="prediction")
    parser.add_argument("--reference_field", default="reference")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?|[^\sA-Za-z0-9]", text.lower())


def safe_meteor(ref_tokens: List[List[str]], pred_tokens: List[str]) -> float:
    try:
        return float(meteor_score(ref_tokens, pred_tokens))
    except LookupError:
        return float("nan")


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    return sum(p.strip() == r.strip() for p, r in zip(predictions, references)) / len(predictions)


def length_stats(texts: Sequence[str]) -> Dict[str, float]:
    lengths = [len(tokenize(t)) for t in texts]
    if not lengths:
        return {"mean": 0.0, "min": 0, "max": 0}
    return {"mean": sum(lengths) / len(lengths), "min": min(lengths), "max": max(lengths)}


def compute_core_metrics(rows: List[Dict], prediction_field: str, reference_field: str) -> Dict:
    filtered = []
    for row in rows:
        pred = row.get(prediction_field)
        ref = row.get(reference_field)
        if pred is None or ref is None:
            continue
        filtered.append((str(pred), str(ref), row.get("task", "unknown")))

    predictions = [x[0] for x in filtered]
    references = [x[1] for x in filtered]
    pred_tokens = [tokenize(x) for x in predictions]
    ref_tokens = [tokenize(x) for x in references]

    metrics: Dict[str, object] = {
        "num_total_rows": len(rows),
        "num_evaluated": len(filtered),
        "exact_match": exact_match(predictions, references),
        "prediction_length_tokens": length_stats(predictions),
        "reference_length_tokens": length_stats(references),
    }
    if not filtered:
        return metrics

    smoothie = SmoothingFunction().method3
    bleu_refs = [[r] for r in ref_tokens]
    metrics.update({
        "bleu": float(corpus_bleu(bleu_refs, pred_tokens, smoothing_function=smoothie)),
        "bleu_1": float(corpus_bleu(bleu_refs, pred_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)),
        "bleu_2": float(corpus_bleu(bleu_refs, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)),
        "bleu_3": float(corpus_bleu(bleu_refs, pred_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie)),
        "bleu_4": float(corpus_bleu(bleu_refs, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)),
    })

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_totals = defaultdict(float)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for name, score in scores.items():
            rouge_totals[f"{name}_precision"] += score.precision
            rouge_totals[f"{name}_recall"] += score.recall
            rouge_totals[f"{name}_fmeasure"] += score.fmeasure
    for key, value in rouge_totals.items():
        metrics[key] = float(value / len(filtered))

    meteor_values = [safe_meteor([r], p) for r, p in zip(ref_tokens, pred_tokens)]
    valid_meteor = [x for x in meteor_values if not math.isnan(x)]
    metrics["meteor"] = float(sum(valid_meteor) / len(valid_meteor)) if valid_meteor else None
    metrics["meteor_available"] = len(valid_meteor) == len(meteor_values)
    return metrics


def compute_metrics(rows: List[Dict], prediction_field: str, reference_field: str) -> Dict:
    metrics = compute_core_metrics(rows, prediction_field, reference_field)
    by_task = {}
    for task in sorted({row.get("task", "unknown") for row in rows}):
        task_rows = [row for row in rows if row.get("task", "unknown") == task]
        by_task[task] = compute_core_metrics(task_rows, prediction_field, reference_field)
    metrics["by_task"] = by_task
    return metrics


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.predictions_jsonl)
    metrics = compute_metrics(rows, args.prediction_field, args.reference_field)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
