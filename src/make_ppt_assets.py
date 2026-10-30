"""Build the extra review figures (metrics table, confusion matrices) and gather every figure into ppt/.

Usage: python -m src.make_ppt_assets
"""
from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs")
PPT = Path("ppt")
CLIENTS = (
    ("hospital_cxr_montgomery", "CXR Montgomery"),
    ("hospital_cxr_shenzhen", "CXR Shenzhen"),
    ("hospital_clinical_a", "Clinical A"),
    ("hospital_clinical_b", "Clinical B"),
    ("hospital_ct", "CT"),
)
METHODS = (
    ("modality_aware_fedavg", "Modality-aware FedAvg (ours)"),
    ("plain_fedavg", "FedAvg, unweighted"),
    ("local_only", "Local-only"),
    ("centralized", "Centralized (not private)"),
)
METRICS = (
    ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("balanced_accuracy", "Balanced acc."),
    ("auroc", "AUROC"), ("auprc", "AUPRC"), ("f1", "F1"),
)


def metrics_table(seeds) -> None:
    rows = []
    for key, label in METHODS:
        row = [label]
        for metric, _ in METRICS:
            per_seed = [
                np.nanmean([s["methods"][key]["test_per_client"][c][metric] or np.nan for c, _ in CLIENTS])
                for s in seeds
            ]
            row.append(f"{np.mean(per_seed):.3f} ± {np.std(per_seed):.3f}")
        rows.append(row)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Method"] + [m[1] for m in METRICS], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for column in range(len(METRICS) + 1):
        table[0, column].set_facecolor("#1f4a7c")
        table[0, column].set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Held-out test metrics, mean of 5 clients (mean ± std, {len(seeds)} seeds)", fontsize=10)
    plt.tight_layout()
    plt.savefig(PPT / "metrics_table.png", dpi=170)
    plt.close()


def confusion_matrices(seed_result) -> None:
    run = seed_result["methods"]["modality_aware_fedavg"]["test_per_client"]
    fig, axes = plt.subplots(1, len(CLIENTS), figsize=(14, 3.2))
    for ax, (client, label) in zip(axes, CLIENTS):
        d = run[client]
        pos = d["positive_count"]
        neg = d["n"] - pos
        tp, tn = round(d["sensitivity"] * pos), round(d["specificity"] * neg)
        matrix = np.array([[tn, neg - tn], [pos - tp, tp]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13, color="white" if value > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred neg", "Pred pos"], fontsize=8)
        ax.set_yticks([0, 1], ["True neg", "True pos"], fontsize=8)
        ax.set_title(f"{label} (n={d['n']})", fontsize=9)
    fig.suptitle(f"Confusion matrices, modality-aware FedAvg, seed {seed_result['seed']} (held-out test)", fontsize=10)
    plt.tight_layout()
    plt.savefig(PPT / "confusion_matrices.png", dpi=170)
    plt.close()


def main() -> None:
    PPT.mkdir(exist_ok=True)
    seeds = [json.load(open(f)) for f in sorted(glob.glob(str(OUT / "study" / "seed_*.json")))]
    metrics_table(seeds)
    confusion_matrices(seeds[0])
    for path in (OUT / "figures").glob("*.png"):
        shutil.copy(path, PPT / path.name)
    for name in ("study_summary.csv",):
        shutil.copy(OUT / name, PPT / name)
    for path in sorted(PPT.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
