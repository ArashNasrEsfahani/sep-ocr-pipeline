import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from datasets import load_dataset  # type: ignore[import]
from Levenshtein import distance as levenshtein_distance  # type: ignore[import]
from paddleocr import PaddleOCR
from PIL import Image
from tqdm import tqdm

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
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = _remove_tatweel(text)
    text = text.translate(TRANSLATION_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_split(dataset_name: str, split: str, limit: Optional[int]) -> Tuple[List[Dict], Dict[str, str]]:
    ds = load_dataset(dataset_name, split=split)
    if limit is not None:
        limit = min(limit, len(ds))
        ds = ds.select(range(limit))
    samples = []
    meta = {"dataset": dataset_name, "split": split, "total": len(ds)}
    for idx, sample in enumerate(ds):
        samples.append({
            "index": idx,
            "image": sample["image_path"],
            "text": sample["text"],
        })
    return samples, meta


def read_image(source) -> np.ndarray:
    if isinstance(source, Image.Image):
        rgb = source.convert("RGB")
        arr = np.array(rgb)
    else:
        with Image.open(source) as img:
            rgb = img.convert("RGB")
            arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def recognize_lines(ocr: PaddleOCR, samples: List[Dict], args: argparse.Namespace) -> Tuple[List[Dict], List[str], List[str], List[float], float]:
    records: List[Dict] = []
    references: List[str] = []
    predictions: List[str] = []
    confidences: List[float] = []
    total_time = 0.0

    iterator = samples
    for sample in tqdm(iterator, desc="recognition", unit="line"):
        reference = normalize_text(sample["text"])
        image = read_image(sample["image"])

        start = time.perf_counter()
        rec_result = ocr.ocr(image, det=False, rec=True, cls=args.use_angle_cls)
        total_time += time.perf_counter() - start

        if rec_result and len(rec_result[0]) > 0:
            pred_text, score = rec_result[0][0]
            score = float(score)
        else:
            pred_text = ""
            score = 0.0

        pred_text = normalize_text(pred_text)
        if args.reverse_prediction:
            pred_text = pred_text[::-1]

        image_hint = getattr(sample["image"], "filename", "") if isinstance(sample["image"], Image.Image) else str(sample["image"])
        records.append({
            "dataset_index": sample["index"],
            "image_hint": image_hint,
            "reference": reference,
            "prediction": pred_text,
            "confidence": score,
        })
        references.append(reference)
        predictions.append(pred_text)
        confidences.append(score)

    return records, references, predictions, confidences, total_time


def _word_tokens(text: str) -> List[str]:
    return [tok for tok in text.split() if tok]


def _token_edit_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    m, n = len(ref_tokens), len(hyp_tokens)
    if m == 0:
        return n
    if n == 0:
        return m

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n]


def compute_metrics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    pairs = [(ref, hyp) for ref, hyp in zip(references, predictions) if len(ref) > 0]
    empty_refs = len(references) - len(pairs)

    if pairs:
        char_errors = sum(levenshtein_distance(ref, hyp) for ref, hyp in pairs)
        char_total = sum(len(ref) for ref, _ in pairs)
        cer = (char_errors / char_total) if char_total else 0.0

        word_errors = 0
        word_total = 0
        for ref, hyp in pairs:
            ref_tokens = _word_tokens(ref)
            if not ref_tokens:
                continue
            hyp_tokens = _word_tokens(hyp)
            word_errors += _token_edit_distance(ref_tokens, hyp_tokens)
            word_total += len(ref_tokens)
        wer = (word_errors / word_total) if word_total else 0.0
    else:
        char_errors = 0
        char_total = 0
        cer = 0.0
        word_errors = 0
        word_total = 0
        wer = 0.0

    return {
        "char_errors": float(char_errors),
        "char_total": float(char_total),
        "cer": float(cer),
        "word_errors": float(word_errors),
        "word_total": float(word_total),
        "wer": float(wer),
        "empty_reference_count": int(empty_refs),
    }


def save_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaddleOCR recognition on Parsynth OCR dataset")
    parser.add_argument("--dataset", default="hezarai/parsynth-ocr-200k", help="HuggingFace dataset identifier")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, help="Optional limit for number of samples")
    parser.add_argument("--pred-out", type=Path, required=True, help="Destination JSONL for predictions")
    parser.add_argument("--metrics-out", type=Path, required=True, help="Destination JSON metrics file")
    parser.add_argument("--lang", default="fa", help="Language flag for PaddleOCR recognition model")
    parser.add_argument("--use-angle-cls", action="store_true", help="Enable angle classifier during recognition")
    parser.add_argument("--rec-image-shape", default="3, 48, 320", help="Override PaddleOCR rec_image_shape (e.g. '3, 64, 640')")
    parser.add_argument("--reverse-prediction", action="store_true", help="Reverse model outputs (useful for RTL languages)")
    args = parser.parse_args()

    samples, meta = load_split(args.dataset, args.split, args.limit)

    ocr = PaddleOCR(
        det=False,
        rec=True,
        lang=args.lang,
        use_angle_cls=args.use_angle_cls,
        rec_image_shape=args.rec_image_shape,
        show_log=False,
    )

    records, references, predictions, confidences, total_time = recognize_lines(ocr, samples, args)

    metrics = compute_metrics(references, predictions)
    metrics.update({
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(records),
        "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "inference_time_sec": float(total_time),
        "samples_per_sec": float(len(records) / total_time) if total_time else 0.0,
    })
    metrics.update(meta)

    save_jsonl(records, args.pred_out)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
