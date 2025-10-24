import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset  # type: ignore[import]
from Levenshtein import distance as levenshtein_distance  # type: ignore[import]


def load_predictions(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def rank_errors(records: List[Dict]) -> List[Dict]:
    ranked = []
    for record in records:
        ref = record.get("reference", "")
        pred = record.get("prediction", "")
        char_err = levenshtein_distance(ref, pred)
        ranked.append({
            **record,
            "char_error": int(char_err),
            "char_length": len(ref),
            "cer_local": (char_err / len(ref)) if len(ref) else 0.0,
        })
    ranked.sort(key=lambda x: x["char_error"], reverse=True)
    return ranked


def export_examples(ranked: List[Dict], top_k: int, out_dir: Path, copy_images: bool, dataset=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "error_report.json"
    summary = ranked[:top_k]

    if copy_images:
        for idx, item in enumerate(summary, start=1):
            src_img = None
            if dataset is not None:
                ds_index = item.get("dataset_index")
                if ds_index is not None and 0 <= ds_index < len(dataset):
                    src_img = dataset[int(ds_index)]["image_path"]
            if src_img is not None:
                dst = out_dir / f"error_{idx:02d}.png"
                try:
                    src_img.save(dst)
                    item["copied_image"] = str(dst)
                except OSError as exc:
                    item["copied_image"] = f"save_failed: {exc}"
            else:
                item["copied_image"] = "missing"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved top-{top_k} error report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OCR recognition errors")
    parser.add_argument("--pred", type=Path, required=True, help="Predictions JSONL from run_recognition.py")
    parser.add_argument("--top-k", type=int, default=5, help="Number of examples to export")
    parser.add_argument("--out-dir", type=Path, required=True, help="Destination directory for error artifacts")
    parser.add_argument("--copy-images", action="store_true", help="Copy original images alongside the report")
    parser.add_argument("--dataset", default="hezarai/parsynth-ocr-200k", help="Dataset identifier to reload images")
    parser.add_argument("--split", default="test", help="Dataset split corresponding to predictions")
    parser.add_argument("--limit", type=int, help="Optional limit to match recognition subset")
    args = parser.parse_args()

    records = load_predictions(args.pred)
    ranked = rank_errors(records)

    dataset = None
    if args.copy_images:
        dataset = load_dataset(args.dataset, split=args.split)
        if args.limit is not None:
            dataset = dataset.select(range(min(args.limit, len(dataset))))

    export_examples(ranked, args.top_k, args.out_dir, args.copy_images, dataset)


if __name__ == "__main__":
    main()
