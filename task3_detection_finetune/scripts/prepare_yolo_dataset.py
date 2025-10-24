import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def _load_annotations(path: Path) -> Dict[str, List[List[float]]]:
    records: Dict[str, List[List[float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            records[sample["filename"]] = sample["boxes"]
    return records


def _xywh_from_polygon(poly: List[float]) -> Tuple[float, float, float, float]:
    xs = poly[0::2]
    ys = poly[1::2]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return x_min, y_min, x_max, y_max


def _write_label(out_file: Path, boxes: List[List[float]], width: int, height: int) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as writer:
        for polygon in boxes:
            x_min, y_min, x_max, y_max = _xywh_from_polygon(polygon)
            cx = ((x_min + x_max) / 2.0) / width
            cy = ((y_min + y_max) / 2.0) / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height
            writer.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def _copy_image(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _split_filenames(filenames: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = filenames[:]
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    return train, val


def _write_dataset_yaml(path: Path, root: Path) -> None:
    content = (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: {0: 'textline'}\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO-format dataset from Arshasb annotations")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing source images")
    parser.add_argument("--annotations", type=Path, required=True, help="JSONL file with quadrilateral boxes")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of samples to reserve for validation")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for splitting")
    args = parser.parse_args()

    annotations = _load_annotations(args.annotations)
    available = [name for name in annotations if (args.images / name).exists()]
    if not available:
        raise RuntimeError("No matching images found for provided annotations")

    train_split, val_split = _split_filenames(available, args.val_ratio, args.seed)

    for subset, subset_names in {"train": train_split, "val": val_split}.items():
        for filename in subset_names:
            src_img = args.images / filename
            with Image.open(src_img) as img:
                width, height = img.width, img.height
            label_path = args.out / "labels" / subset / f"{Path(filename).stem}.txt"
            _write_label(label_path, annotations[filename], width, height)
            dst_img = args.out / "images" / subset / filename
            _copy_image(src_img, dst_img)

    dataset_yaml = args.out / "arshasb.yaml"
    _write_dataset_yaml(dataset_yaml, args.out)
    summary = {
        "train": len(train_split),
        "val": len(val_split),
        "dataset_yaml": str(dataset_yaml),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
