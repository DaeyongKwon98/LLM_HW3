import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def best_eval_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    data = read_json(path)
    return data.get("best_eval_metrics", data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize mDPO LoRA sweep metrics.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    rows = []
    with Path(args.manifest_csv).open("r", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            metrics_path = Path(item["metrics_json"])
            best_metrics_path = Path(item["best_metrics_json"])
            if not metrics_path.exists():
                row = dict(item)
                row["status"] = "missing_metrics"
                rows.append(row)
                continue

            metrics = read_json(metrics_path)
            best = best_eval_metrics(best_metrics_path)
            by_task = metrics.get("by_task", {})
            logp = by_task.get("logp", {})
            qm9 = by_task.get("qm9", {})
            row = {
                **item,
                "status": "ok",
                "test_n": metrics.get("num_evaluated"),
                "test_bleu1": metrics.get("bleu_1"),
                "test_bleu4": metrics.get("bleu_4"),
                "test_rouge1": metrics.get("rouge1_fmeasure"),
                "test_rouge2": metrics.get("rouge2_fmeasure"),
                "test_rougeL": metrics.get("rougeL_fmeasure"),
                "test_meteor": metrics.get("meteor"),
                "logp_bleu4": logp.get("bleu_4"),
                "logp_rougeL": logp.get("rougeL_fmeasure"),
                "logp_meteor": logp.get("meteor"),
                "qm9_bleu4": qm9.get("bleu_4"),
                "qm9_rougeL": qm9.get("rougeL_fmeasure"),
                "qm9_meteor": qm9.get("meteor"),
                "val_eval_loss": best.get("eval_loss"),
                "val_sft_loss": best.get("eval_loss_sft"),
                "val_copo_loss": best.get("eval_loss_copo"),
                "val_accuracy": best.get("eval_accuracy"),
                "val_reward_margin": best.get("eval_reward_margin_mean"),
            }
            rows.append(row)

    def sort_key(row: dict) -> tuple:
        status = 0 if row.get("status") == "ok" else 1
        meteor = row.get("test_meteor")
        try:
            meteor_value = float(meteor)
        except (TypeError, ValueError):
            meteor_value = -1.0
        return (status, -meteor_value)

    rows.sort(key=sort_key)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")
    for row in rows:
        if row.get("status") != "ok":
            print(f"{row.get('run_name')}: {row.get('status')}")
            continue
        print(
            "{run_name}: METEOR={meteor:.6f} BLEU4={bleu4:.6f} ROUGE-L={rougel:.6f} "
            "val_loss={val_loss:.6f} val_acc={val_acc:.6f}".format(
                run_name=row["run_name"],
                meteor=float(row["test_meteor"]),
                bleu4=float(row["test_bleu4"]),
                rougel=float(row["test_rougeL"]),
                val_loss=float(row["val_eval_loss"]),
                val_acc=float(row["val_accuracy"]),
            )
        )


if __name__ == "__main__":
    main()
