import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore[import]

try:  # allowlist Ultralytics DetectionModel for torch.load safety checks
    from torch.serialization import add_safe_globals  # type: ignore[attr-defined]
    from ultralytics.nn.tasks import DetectionModel  # type: ignore[import]

    add_safe_globals([DetectionModel])  # type: ignore[misc]
except Exception:
    pass


def _patch_torch_load() -> None:
    original_load = torch.load

    def _wrapped(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _wrapped  # type: ignore[assignment]


def _xyxy_to_polygon(x1: float, y1: float, x2: float, y2: float) -> List[float]:
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _iter_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference and export JSONL detections")
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO weights (*.pt)")
    parser.add_argument("--images", type=Path, required=True, help="Directory of images to run inference on")
    parser.add_argument("--pred-out", type=Path, required=True, help="Output JSONL file for detections")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="auto", help="Compute device to use")
    parser.add_argument("--limit", type=int, help="Optional limit on number of images")
    args = parser.parse_args()

    _patch_torch_load()
    model = YOLO(str(args.weights))
    image_paths = _iter_images(args.images)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    records = []
    start = time.perf_counter()
    for img_path in tqdm(image_paths, desc="yolo_infer"):
        result = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            verbose=False,
        )[0]
        boxes = result.boxes
        polygons: List[List[float]] = []
        scores: List[float] = []
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
            for (x1, y1, x2, y2), score in zip(xyxy, conf):
                polygons.append(_xyxy_to_polygon(float(x1), float(y1), float(x2), float(y2)))
                scores.append(float(score))

        records.append({
            "filename": img_path.name,
            "boxes": polygons,
            "confidence": scores,
            "source": "yolov8_finetune",
        })
    elapsed = time.perf_counter() - start

    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    with args.pred_out.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "images": len(image_paths),
        "elapsed_sec": elapsed,
        "fps": len(image_paths) / elapsed if elapsed > 0 else 0.0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
