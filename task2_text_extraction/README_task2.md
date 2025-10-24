# Task 2 – Text Extraction with Pre-trained OCR

This directory contains the recognition-only experiments for the Parsynth-OCR dataset using PaddleOCR.

## Layout

- `scripts/`
  - `run_recognition.py` – run inference on Parsynth splits and generate predictions + metrics.
  - `analyze_errors.py` – surface error examples for qualitative review.
- `artifacts/`
  - `recognition_predictions.jsonl` – latest prediction dump (id, text, prediction, score).
  - `metrics.json` – CER/WER summary produced by `run_recognition.py`.
  - `error_cases/` – illustrative misrecognitions exported as PNGs and a JSON report.
- `requirements_task2.txt` – Python packages required beyond Task 1.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_task2.txt
python scripts/run_recognition.py --split test --limit 500 --pred-out artifacts/recognition_predictions.jsonl --metrics-out artifacts/metrics.json --reverse-prediction
python scripts/analyze_errors.py --pred artifacts/recognition_predictions.jsonl --top-k 5 --out-dir artifacts/error_cases --copy-images --split test --limit 500
```

## Current Status

- Model: PaddleOCR PP-OCRv4 recognition head (`lang=fa`), detection disabled (inputs are cropped lines). Predicted strings are reversed to compensate for RTL rendering.
- Dataset: `hezarai/parsynth-ocr-200k` (HuggingFace). Using the `test` split for evaluation; optionally sub-sampled via `--limit`.
- Metrics: Character Error Rate (CER) and Word Error Rate (WER) computed via `Levenshtein`-based character edit distance and custom token-level WER.
- Error analysis: Reports highest CER cases with image dumps for inspection.

## Metrics (latest run, 500-sample subset)

- CER: 0.4183 (char_errors 3390 / char_total 8105)
- WER: 0.7231 (word_errors 1115 / word_total 1542)
- Avg confidence: 0.81, throughput: 33.7 lines/s (CPU)
- Empty-reference samples skipped: 1

## Notes

- Parsynth samples are synthetic Persian lines; outputs tend to include diacritics. Normalization uses simple Unicode NFC to align with labels.
- For more robust evaluation, consider complementing with a small manually transcribed set of real receipts.
