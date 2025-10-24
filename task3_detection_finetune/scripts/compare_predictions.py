import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from shapely.geometry import Polygon  # type: ignore[import]


@dataclass
class SampleStats:
    tp: int
    fp: int
    fn: int

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if p + r else 0.0


def _load_jsonl(path: Path) -> Dict[str, Dict]:
    data: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            data[record["filename"]] = record
    return data


def _polygon_from_flat(coords: Iterable[float]) -> Polygon:
    pts = list(coords)
    if len(pts) != 8:
        raise ValueError("Each box must have eight coordinates")
    poly = Polygon([(pts[i], pts[i + 1]) for i in range(0, 8, 2)])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _match_sample(pred_boxes: List[List[float]], confs: List[float], gt_boxes: List[List[float]], iou_thresh: float) -> SampleStats:
    pred_polys = [_polygon_from_flat(box) for box in pred_boxes]
    gt_polys = [_polygon_from_flat(box) for box in gt_boxes]
    if not gt_polys:
        return SampleStats(tp=0, fp=len(pred_polys), fn=0)

    matched_gt = set()
    tp = 0
    order = sorted(range(len(pred_polys)), key=lambda idx: confs[idx] if idx < len(confs) else 1.0, reverse=True)
    for idx in order:
        pred_poly = pred_polys[idx]
        best_iou = 0.0
        best_gt = -1
        for gt_idx, gt_poly in enumerate(gt_polys):
            if gt_idx in matched_gt:
                continue
            inter = pred_poly.intersection(gt_poly).area
            if inter <= 0.0:
                continue
            union = pred_poly.union(gt_poly).area
            if union <= 0.0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_iou >= iou_thresh and best_gt >= 0:
            matched_gt.add(best_gt)
            tp += 1

    fp = len(pred_polys) - tp
    fn = len(gt_polys) - len(matched_gt)
    return SampleStats(tp=tp, fp=fp, fn=fn)


def evaluate(pred_path: Path, gt_path: Path, iou_thresh: float) -> Dict[str, SampleStats]:
    preds = _load_jsonl(pred_path)
    gts = _load_jsonl(gt_path)

    filenames = sorted(set(preds.keys()) | set(gts.keys()))
    stats: Dict[str, SampleStats] = {}
    for filename in filenames:
        pred = preds.get(filename, {})
        gt = gts.get(filename, {})
        boxes = pred.get("boxes", [])
        confs = pred.get("confidence", [1.0] * len(boxes))
        gt_boxes = gt.get("boxes", [])
        stats[filename] = _match_sample(boxes, confs, gt_boxes, iou_thresh)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and fine-tuned detection runs per sample")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline predictions JSONL")
    parser.add_argument("--finetuned", type=Path, required=True, help="Fine-tuned predictions JSONL")
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth JSONL")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--top-k", type=int, default=5, help="Number of most improved samples to show")
    args = parser.parse_args()

    base_stats = evaluate(args.baseline, args.gt, args.iou)
    ft_stats = evaluate(args.finetuned, args.gt, args.iou)

    improvements: List[Tuple[float, str]] = []
    for filename in base_stats:
        delta_recall = ft_stats[filename].recall() - base_stats[filename].recall()
        delta_precision = ft_stats[filename].precision() - base_stats[filename].precision()
        score = delta_recall + 0.5 * delta_precision
        improvements.append((score, filename))

    improvements.sort(reverse=True)
    print("filename,base_recall,ft_recall,base_prec,ft_prec,base_fn,ft_fn")
    for _, filename in improvements[: args.top_k]:
        b = base_stats[filename]
        f = ft_stats[filename]
        print(
            f"{filename},{b.recall():.3f},{f.recall():.3f},{b.precision():.3f},{f.precision():.3f},{b.fn},{f.fn}"
        )


if __name__ == "__main__":
    main()
