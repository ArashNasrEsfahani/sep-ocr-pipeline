import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["overall"] if "overall" in data else data


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline vs fine-tuned detection metrics")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline metrics JSON path")
    parser.add_argument("--finetuned", type=Path, required=True, help="Fine-tuned metrics JSON path")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    args = parser.parse_args()

    baseline = _load_metrics(args.baseline)
    finetuned = _load_metrics(args.finetuned)

    labels = ["Precision", "Recall", "F1"]
    baseline_vals = [baseline.get("precision", 0.0), baseline.get("recall", 0.0), baseline.get("f1", 0.0)]
    finetuned_vals = [finetuned.get("precision", 0.0), finetuned.get("recall", 0.0), finetuned.get("f1", 0.0)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline")
    ax.bar(x + width / 2, finetuned_vals, width, label="Fine-tuned")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Arshasb Detection Metrics")
    ax.legend()
    for xpos, val in zip(x - width / 2, baseline_vals):
        ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for xpos, val in zip(x + width / 2, finetuned_vals):
        ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
