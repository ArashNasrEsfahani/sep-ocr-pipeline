import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd

_COORD_RE = re.compile(r"\(([-+]?\d+),\s*([-+]?\d+)\)")


def _points_from_row(row: pd.Series) -> List[int]:
    points = {}
    for key in ("point1", "point2", "point3", "point4"):
        match = _COORD_RE.fullmatch(str(row[key]).strip())
        if not match:
            raise ValueError(f"Failed to parse coordinate '{row[key]}'")
        points[key] = (int(match.group(1)), int(match.group(2)))

    ordered_keys = ("point1", "point3", "point4", "point2")
    coords: List[int] = []
    for key in ordered_keys:
        coords.extend(points[key])
    return coords


def _load_lines(xlsx_path: Path) -> List[List[int]]:
    df = pd.read_excel(xlsx_path)
    boxes: List[List[int]] = []
    for _, row in df.iterrows():
        boxes.append(_points_from_row(row))
    return boxes


def build_subset(source_root: Path, out_images: Path, out_jsonl: Path, sample_size: int, seed: int) -> None:
    rng = random.Random(seed)
    available = sorted([p for p in source_root.iterdir() if p.is_dir()])
    if sample_size and sample_size < len(available):
        selected = rng.sample(available, sample_size)
    else:
        selected = available

    out_images.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for idx, folder in enumerate(selected):
        image_path = folder / f"page_{folder.name}.png"
        xlsx_path = folder / f"line_{folder.name}.xlsx"
        if not image_path.exists() or not xlsx_path.exists():
            print(f"[skip] Missing image or annotation in {folder}")
            continue

        target_name = f"arshasb_{folder.name}.png"
        target_path = out_images / target_name
        shutil.copy(image_path, target_path)

        boxes = _load_lines(xlsx_path)
        records.append({
            "filename": target_name,
            "boxes": boxes,
            "confidence": [1.0] * len(boxes),
            "source": "arshasb",
        })

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1} items")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} ground-truth items to {out_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a subset of the Arshasb dataset for detection experiments.")
    parser.add_argument("--source-root", type=Path, required=True, help="Path to Arshasb_7k root directory")
    parser.add_argument("--out-images", type=Path, required=True, help="Destination directory for copied images")
    parser.add_argument("--out-annotations", type=Path, required=True, help="Output JSONL annotation file")
    parser.add_argument("--sample-size", type=int, default=150, help="Number of documents to sample")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for sampling")

    args = parser.parse_args()
    build_subset(args.source_root, args.out_images, args.out_annotations, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
