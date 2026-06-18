#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


COUNT_BLOCK_RE = re.compile(
    r"The SMILES (?:shows|indicates) (?P<body>.*?)(?:\.\s|;|\.$)",
    re.IGNORECASE,
)
COUNT_ITEM_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<label>[A-Za-z][A-Za-z -]*?)(?:\s+atom\(s\)|\s+group\(s\)|\s+bond\(s\)|\s+ring\(s\)|(?=,|\.|;|\band\b|$))",
    re.IGNORECASE,
)
HBD_HBA_RE = re.compile(r"HBD/HBA\s*=\s*(?P<hbd>\d+)\s*/\s*(?P<hba>\d+)", re.IGNORECASE)
TPSA_RE = re.compile(r"TPSA\s+(?P<tpsa>-?\d+(?:\.\d+)?)\s*A\^?2", re.IGNORECASE)


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_label(label: str) -> str:
    label = label.lower().strip()
    label = re.sub(r"\s+", " ", label)
    return label


def parse_counts(text: str) -> Optional[Dict[str, int]]:
    match = COUNT_BLOCK_RE.search(text)
    if not match:
        return None
    body = match.group("body")
    counts: Dict[str, int] = {}
    for item in COUNT_ITEM_RE.finditer(body):
        label = norm_label(item.group("label"))
        counts[label] = int(item.group("count"))
    return counts or None


def parse_hbd_hba(text: str) -> Optional[Tuple[int, int]]:
    match = HBD_HBA_RE.search(text)
    if not match:
        return None
    return int(match.group("hbd")), int(match.group("hba"))


def parse_tpsa(text: str) -> Optional[float]:
    match = TPSA_RE.search(text)
    if not match:
        return None
    return float(match.group("tpsa"))


def mean(values: Iterable[bool]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def summarize(rows: List[Dict], tpsa_tol: float) -> Dict:
    atom_ok = []
    hbd_hba_ok = []
    tpsa_ok = []
    parse_failures = defaultdict(int)

    for row in rows:
        pred = str(row.get("prediction", ""))
        ref = str(row.get("reference", ""))

        pred_counts = parse_counts(pred)
        ref_counts = parse_counts(ref)
        pred_hbd_hba = parse_hbd_hba(pred)
        ref_hbd_hba = parse_hbd_hba(ref)
        pred_tpsa = parse_tpsa(pred)
        ref_tpsa = parse_tpsa(ref)

        if pred_counts is None:
            parse_failures["prediction_counts"] += 1
        if ref_counts is None:
            parse_failures["reference_counts"] += 1
        if pred_hbd_hba is None:
            parse_failures["prediction_hbd_hba"] += 1
        if ref_hbd_hba is None:
            parse_failures["reference_hbd_hba"] += 1
        if pred_tpsa is None:
            parse_failures["prediction_tpsa"] += 1
        if ref_tpsa is None:
            parse_failures["reference_tpsa"] += 1

        atom_ok.append(pred_counts is not None and ref_counts is not None and pred_counts == ref_counts)
        hbd_hba_ok.append(pred_hbd_hba == ref_hbd_hba)
        if pred_tpsa is None or ref_tpsa is None:
            tpsa_ok.append(pred_tpsa == ref_tpsa)
        else:
            tpsa_ok.append(abs(pred_tpsa - ref_tpsa) <= tpsa_tol)

    return {
        "num_total_rows": len(rows),
        "atom_count_acc": mean(atom_ok),
        "hbd_hba_acc": mean(hbd_hba_ok),
        "tpsa_acc": mean(tpsa_ok),
        "tpsa_tolerance": tpsa_tol,
        "parse_failures": dict(parse_failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structured property values extracted from captions.")
    parser.add_argument("--predictions_jsonl", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--tpsa_tol", type=float, default=0.05, help="Absolute TPSA tolerance for exact one-decimal matches.")
    args = parser.parse_args()

    rows = load_jsonl(args.predictions_jsonl)
    metrics = summarize(rows, args.tpsa_tol)
    by_task = {}
    for task in sorted({row.get("task", "unknown") for row in rows}):
        by_task[task] = summarize([row for row in rows if row.get("task", "unknown") == task], args.tpsa_tol)
    metrics["by_task"] = by_task

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
