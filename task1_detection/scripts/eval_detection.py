import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from shapely.geometry import Polygon


@dataclass
class EvalStats:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def update(self, tp: int, fp: int, fn: int) -> None:
        self.true_positives += tp
        self.false_positives += fp
        self.false_negatives += fn

    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if p + r else 0.0


def _polygon_from_list(coords: Iterable[float]) -> Polygon:
    pts = list(coords)
    if len(pts) != 8:
        raise ValueError("Each box must contain 8 values (x1,y1,...,x4,y4)")
    poly = Polygon([(pts[i], pts[i + 1]) for i in range(0, 8, 2)])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _compute_iou(pred: Polygon, gt: Polygon) -> float:
    if pred.is_empty or gt.is_empty:
        return 0.0
    inter = pred.intersection(gt).area
    if inter <= 0.0:
        return 0.0
    union = pred.union(gt).area
    return inter / union if union > 0.0 else 0.0


def _load_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def evaluate(pred_path: Path, gt_path: Path, iou_thresh: float) -> Dict[str, Dict[str, float]]:
    pred_records = _load_jsonl(pred_path)
    gt_records = _load_jsonl(gt_path)

    gt_map: Dict[str, Dict[str, List]] = {}
    for record in gt_records:
        gt_map[record["filename"]] = record

    pred_map: Dict[str, Dict[str, List]] = {}
    for record in pred_records:
        pred_map[record["filename"]] = record

    stats_by_source: Dict[str, EvalStats] = defaultdict(EvalStats)
    overall = EvalStats()

    filenames = set(gt_map.keys()) | set(pred_map.keys())
    for filename in sorted(filenames):
        pred_entry = pred_map.get(filename, {})
        gt_entry = gt_map.get(filename, {})

        boxes = pred_entry.get("boxes", [])
        confs = pred_entry.get("confidence", [1.0] * len(boxes))
        gt_boxes = gt_entry.get("boxes", [])
        source = gt_entry.get("source", "unknown")

        tp, fp, fn = _match_sample(boxes, confs, gt_boxes, iou_thresh)
        overall.update(tp, fp, fn)
        stats_by_source[source].update(tp, fp, fn)

    print("=== Detection Metrics (IoU>{:.2f}) ===".format(iou_thresh))
    print("Overall  : P={:.3f} R={:.3f} F1={:.3f}".format(overall.precision(), overall.recall(), overall.f1()))
    for source, stats in sorted(stats_by_source.items()):
        print("{:8s}: P={:.3f} R={:.3f} F1={:.3f}".format(source, stats.precision(), stats.recall(), stats.f1()))

    def _pack(stats: EvalStats) -> Dict[str, float]:
        return {
            "precision": stats.precision(),
            "recall": stats.recall(),
            "f1": stats.f1(),
            "tp": stats.true_positives,
            "fp": stats.false_positives,
            "fn": stats.false_negatives,
        }

    by_source = {source: _pack(stats) for source, stats in stats_by_source.items()}
    return {"overall": _pack(overall), "by_source": by_source}


def _match_sample(pred_boxes: List[List[float]], confs: List[float], gt_boxes: List[List[float]], iou_thresh: float) -> Tuple[int, int, int]:
    pred_polys = [_polygon_from_list(box) for box in pred_boxes]
    gt_polys = [_polygon_from_list(box) for box in gt_boxes]

    if not gt_polys:
        return 0, len(pred_polys), 0

    matched_gt = set()
    tp = 0
    order = sorted(range(len(pred_polys)), key=lambda idx: confs[idx] if idx < len(confs) else 1.0, reverse=True)
    for idx in order:
        poly = pred_polys[idx]
        conf = confs[idx] if idx < len(confs) else 1.0
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_poly in enumerate(gt_polys):
            if gt_idx in matched_gt:
                continue
            iou = _compute_iou(poly, gt_poly)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_polys) - tp
    fn = len(gt_polys) - len(matched_gt)
    return tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text box detections against ground truth")
    parser.add_argument("--pred", type=Path, required=True, help="Predictions JSONL")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth JSONL")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--report", type=Path, help="Optional path to save metrics as JSON")
    args = parser.parse_args()

    metrics = evaluate(args.pred, args.gt, args.iou)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
