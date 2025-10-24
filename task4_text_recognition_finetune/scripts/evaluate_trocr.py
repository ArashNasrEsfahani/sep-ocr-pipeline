import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from Levenshtein import distance as levenshtein_distance  # type: ignore[import]
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TRANSLATION_TABLE = str.maketrans({
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ئ": "ی",
    "أ": "ا",
    "إ": "ا",
    "آ": "آ",
    "ؤ": "و",
    "ة": "ه",
    "ۀ": "ه",
    "ﻻ": "لا",
    "٠": "۰",
    "١": "۱",
    "٢": "۲",
    "٣": "۳",
    "٤": "۴",
    "٥": "۵",
    "٦": "۶",
    "٧": "۷",
    "٨": "۸",
    "٩": "۹",
})


def _remove_tatweel(text: str) -> str:
    return text.replace("ـ", "")


def normalize_text(text: str) -> str:
    text = str(text)
    text = text.translate(TRANSLATION_TABLE)
    text = _remove_tatweel(text)
    text = " ".join(text.split())
    return text.strip()


def _word_edit_distance(ref: List[str], hyp: List[str]) -> int:
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def compute_metrics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    char_total = 0
    char_errors = 0
    word_total = 0
    word_errors = 0
    exact_matches = 0
    for ref, hyp in zip(references, predictions):
        ref_norm = normalize_text(ref)
        hyp_norm = normalize_text(hyp)
        if ref_norm == hyp_norm:
            exact_matches += 1
        char_total += len(ref_norm)
        char_errors += levenshtein_distance(ref_norm, hyp_norm)
        ref_words = [w for w in ref_norm.split() if w]
        hyp_words = [w for w in hyp_norm.split() if w]
        word_total += len(ref_words)
        if ref_words:
            word_errors += _word_edit_distance(ref_words, hyp_words)
    cer = (char_errors / char_total) if char_total else 0.0
    wer = (word_errors / word_total) if word_total else 0.0
    em = exact_matches / len(references) if references else 0.0
    return {
        "char_errors": float(char_errors),
        "char_total": float(char_total),
        "cer": float(cer),
        "word_errors": float(word_errors),
        "word_total": float(word_total),
        "wer": float(wer),
        "exact_match_rate": float(em),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned TrOCR checkpoint on Parsynth OCR")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing the fine-tuned model")
    parser.add_argument("--dataset", default="hezarai/parsynth-ocr-200k", help="Dataset identifier")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, help="Optional limit on number of samples")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64, help="Maximum generation length")
    parser.add_argument("--pred-out", type=Path, required=True, help="Destination JSONL for predictions")
    parser.add_argument("--metrics-out", type=Path, required=True, help="Destination metrics JSON file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    processor = TrOCRProcessor.from_pretrained(str(args.model_dir))
    model = VisionEncoderDecoderModel.from_pretrained(str(args.model_dir))
    model.to(device)
    model.eval()

    dataset = load_dataset(args.dataset, split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    records = []
    references: List[str] = []
    predictions: List[str] = []
    start_time = time.perf_counter()

    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start : start + args.batch_size]
        images = [img.convert("RGB") for img in batch["image_path"]]
        refs = [normalize_text(txt) for txt in batch["text"]]
        with torch.no_grad():
            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(
                pixel_values,
                max_length=args.max_length,
                num_beams=4,
                early_stopping=True,
            )
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for idx, pred in enumerate(preds):
            records.append({
                "dataset_index": int(start + idx),
                "prediction": normalize_text(pred),
                "reference": refs[idx],
            })
            references.append(refs[idx])
            predictions.append(pred)

    elapsed = time.perf_counter() - start_time
    metrics = compute_metrics(references, predictions)
    metrics.update({
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(references),
        "generation_max_length": args.max_length,
        "batch_size": args.batch_size,
        "inference_time_sec": float(elapsed),
        "samples_per_sec": float(len(references) / elapsed) if elapsed > 0 else 0.0,
    })

    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    with args.pred_out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
