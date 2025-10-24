import argparse
import json
from pathlib import Path
from typing import Dict, List

from Levenshtein import distance as levenshtein_distance  # type: ignore[import]

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


def compute_metrics(records: List[Dict]) -> Dict[str, float]:
    char_total = 0
    char_errors = 0
    word_total = 0
    word_errors = 0
    exact_matches = 0
    for record in records:
        ref = normalize_text(record.get("reference", ""))
        hyp = normalize_text(record.get("prediction", ""))
        if ref == "":
            continue
        if ref == hyp:
            exact_matches += 1
        char_total += len(ref)
        char_errors += levenshtein_distance(ref, hyp)
        ref_words = [w for w in ref.split() if w]
        hyp_words = [w for w in hyp.split() if w]
        word_total += len(ref_words)
        if ref_words:
            word_errors += _word_edit_distance(ref_words, hyp_words)
    cer = (char_errors / char_total) if char_total else 0.0
    wer = (word_errors / word_total) if word_total else 0.0
    em = exact_matches / len(records) if records else 0.0
    return {
        "char_errors": float(char_errors),
        "char_total": float(char_total),
        "cer": float(cer),
        "word_errors": float(word_errors),
        "word_total": float(word_total),
        "wer": float(wer),
        "exact_match_rate": float(em),
        "num_samples": len(records),
    }


def load_predictions(path: Path, limit: int | None = None) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CER/WER/EM metrics from prediction JSONL")
    parser.add_argument("--pred-jsonl", type=Path, required=True, help="Prediction JSONL file from recognition pipeline")
    parser.add_argument("--out", type=Path, required=True, help="Destination JSON for metrics")
    parser.add_argument("--limit", type=int, help="Optional limit to mirror inference subset")
    args = parser.parse_args()

    records = load_predictions(args.pred_jsonl, args.limit)
    metrics = compute_metrics(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
