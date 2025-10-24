import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

_COLOR = (0, 255, 0)


def _load_predictions(path: Path) -> Dict[str, Dict[str, List]]:
    data: Dict[str, Dict[str, List]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            data[record["filename"]] = record
    return data


def _draw_boxes(image: np.ndarray, boxes: List[List[float]], confs: List[float], score_threshold: float) -> np.ndarray:
    canvas = image.copy()
    for idx, coords in enumerate(boxes):
        score = confs[idx] if idx < len(confs) else 1.0
        if score < score_threshold:
            continue
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=_COLOR, thickness=2)
        label = f"{score:.2f}"
        cv2.putText(canvas, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def visualize(images_dir: Path, preds_path: Path, out_dir: Path, limit: int, score_threshold: float) -> None:
    preds = _load_predictions(preds_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for filename, record in tqdm(preds.items(), desc="visualize"):
        img_path = images_dir / filename
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        boxes = record.get("boxes", [])
        confs = record.get("confidence", [])
        annotated = _draw_boxes(image, boxes, confs, score_threshold)
        cv2.imwrite(str(out_dir / filename), annotated)
        processed += 1
        if limit and processed >= limit:
            break

    print(f"Saved overlays for {processed} images to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize detection results as image overlays")
    parser.add_argument("--images", type=Path, required=True, help="Directory with original images")
    parser.add_argument("--pred", type=Path, required=True, help="Predictions JSONL file")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for visualizations")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of images to render")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Minimum confidence to display")
    args = parser.parse_args()

    visualize(args.images, args.pred, args.out, args.limit, args.score_threshold)


if __name__ == "__main__":
    main()
