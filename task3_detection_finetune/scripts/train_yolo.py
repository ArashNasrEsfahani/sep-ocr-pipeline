import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
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


def _extract_metrics(results_csv: Path) -> Optional[dict]:
    if not results_csv.exists():
        return None
    df = pd.read_csv(results_csv)
    if df.empty:
        return None
    last = df.iloc[-1]
    metric_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    extracted = {key: float(last[key]) for key in metric_keys if key in last}
    extracted["epoch"] = int(last["epoch"]) if "epoch" in last else len(df) - 1
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Arshasb text-line detection")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="YOLO dataset yaml file")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained YOLO model to start from")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="auto", help="Compute device to use (auto/cpu/0/0,1)")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"), help="YOLO project directory")
    parser.add_argument("--name", default="arshasb_yolov8n_finetune", help="Training run name")
    parser.add_argument("--summary-out", type=Path, help="Optional path to store metrics summary JSON")
    args = parser.parse_args()

    _patch_torch_load()
    model = YOLO(args.model)
    results = model.train(
        data=str(args.dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        verbose=True,
    )

    save_dir = Path(model.trainer.save_dir if hasattr(model, "trainer") else results.save_dir)  # type: ignore[attr-defined]
    results_csv = save_dir / "results.csv"
    summary = {
        "save_dir": str(save_dir),
        "weights": str(save_dir / "weights" / "best.pt"),
    }
    metrics = _extract_metrics(results_csv)
    if metrics:
        summary.update(metrics)

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
