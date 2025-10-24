import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm


def _setup_ocr(args: argparse.Namespace) -> PaddleOCR:
    return PaddleOCR(
        use_angle_cls=False,
        det=True,
        rec=False,
        lang=args.lang,
        det_db_thresh=args.det_db_thresh,
        det_db_box_thresh=args.det_db_box_thresh,
        det_db_unclip_ratio=args.det_db_unclip_ratio,
        rec_score_thresh=0.0,
        show_log=args.verbose,
    )


def _iter_image_paths(images_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for path in sorted(images_dir.rglob("*")):
        if path.suffix.lower() in exts:
            yield path


def _resize_long_edge(image: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_size:
        return image, 1.0
    scale = max_size / long_edge
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def _apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _polygon_area(box: Iterable[Tuple[float, float]]) -> float:
    pts = list(box)
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _scale_box(box: Iterable[Tuple[float, float]], inv_scale: float) -> List[float]:
    scaled: List[float] = []
    for x, y in box:
        scaled.extend([x * inv_scale, y * inv_scale])
    return scaled


def run_inference(args: argparse.Namespace) -> None:
    images_dir = Path(args.images)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ocr = _setup_ocr(args)
    records: List[Dict[str, object]] = []

    for img_path in tqdm(list(_iter_image_paths(images_dir)), desc="inference"):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[warn] Failed to read {img_path}")
            continue

        if args.clahe:
            image = _apply_clahe(image, clip_limit=args.clahe_clip_limit, tile_size=args.clahe_tile_size)

        processed, scale = _resize_long_edge(image, args.max_side)
        try:
            dt_boxes, elapse = ocr.text_detector(processed)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[error] Detector failed on {img_path}: {exc}")
            continue

        boxes: List[List[float]] = []
        scores: List[float] = []

        if dt_boxes is None or len(dt_boxes) == 0:
            records.append({"filename": img_path.name, "boxes": boxes, "confidence": scores})
            continue

        inv_scale = 1.0 / scale if not math.isclose(scale, 0.0) else 1.0
        scale_factor = inv_scale ** 2
        for poly in dt_boxes.tolist():
            area = _polygon_area(poly) * scale_factor
            if area < args.min_area:
                continue
            scaled_box = _scale_box(poly, inv_scale)
            boxes.append(scaled_box)
            scores.append(1.0)

        records.append({
            "filename": img_path.name,
            "boxes": boxes,
            "confidence": scores,
            "source": "paddleocr_ppocrv4_det",
        })

    with output_path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved detections for {len(records)} images to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PaddleOCR detection on a directory of images.")
    parser.add_argument("--images", required=True, help="Directory of input images")
    parser.add_argument("--out", required=True, help="Destination JSONL file")
    parser.add_argument("--max-side", type=int, default=1280, help="Maximum long edge size for inference")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE to improve low-contrast receipts")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--clahe-tile-size", type=int, default=8, help="CLAHE tile grid size")
    parser.add_argument("--min-area", type=float, default=500.0, help="Minimum polygon area to keep")
    parser.add_argument("--det-db-thresh", type=float, default=0.3, help="DB pre-threshold")
    parser.add_argument("--det-db-box-thresh", type=float, default=0.6, help="DB box threshold")
    parser.add_argument("--det-db-unclip-ratio", type=float, default=2.0, help="DB unclip ratio")
    parser.add_argument("--lang", default="en", help="Language model flag for PaddleOCR")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose PaddleOCR logs")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)
