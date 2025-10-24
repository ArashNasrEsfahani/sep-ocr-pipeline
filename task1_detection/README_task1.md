# Task 1 – Text Detection Pipeline

This folder tracks experiments for receipt/document text-box detection.

## Layout

- `data/` – curated datasets
  - `receipts/` – POS receipt photos + manual annotations (pending manual review)
  - `arshasb/` – Arshasb Persian documents subset with line-level boxes
- `scripts/` – runnable CLI utilities
- `artifacts/` – generated metrics, plots, detections
- `requirements_task1.txt` – Python dependencies for this task

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_task1.txt
python scripts/prepare_arshasb_subset.py \
  --source-root data/arshasb/Arshasb_7k \
  --out-images data/arshasb/images \
  --out-annotations data/arshasb/annotations/ground_truth.jsonl \
  --sample-size 150
python scripts/run_detection.py --images data/arshasb/images --out artifacts/arshasb_detections.jsonl --max-side 1536
python scripts/run_detection.py --images data/receipts/images --out artifacts/receipts_detections.jsonl --clahe
python scripts/eval_detection.py --pred artifacts/arshasb_detections.jsonl --gt data/arshasb/annotations/ground_truth.jsonl --iou 0.5 --report artifacts/metrics_arshasb.json
python scripts/visualize_results.py --images data/arshasb/images --pred artifacts/arshasb_detections.jsonl --out artifacts/plots/arshasb --limit 4
python scripts/visualize_results.py --images data/receipts/images --pred artifacts/receipts_detections.jsonl --out artifacts/plots/receipts --limit 4
```

The combined inference output required for reporting is written to `artifacts/detections.jsonl`.

## Current Metrics

- Arshasb subset (150 docs, IoU>0.5): Precision 0.793 / Recall 0.801 / F1 0.797 (`artifacts/metrics_arshasb.json`).
- Receipt evaluation pending (manual annotation template in `data/receipts/annotations/ground_truth_template.jsonl`).

## Notes

- Pre-trained detector: PaddleOCR PP-OCRv4 (DB++)
- Evaluation metric: Precision/Recall/F1 @ IoU>0.5 on quadrilateral boxes
- Visual inspection covers 3–4 representative images per domain
