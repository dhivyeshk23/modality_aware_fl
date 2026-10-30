"""Draw the project architecture diagram to ppt/architecture.png.

Usage: python -m src.make_architecture
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLORS = {
    "data": ("#dbe9f7", "#2b6cb0"),
    "enc": ("#dff3e4", "#2f855a"),
    "fusion": ("#fdebd3", "#c05621"),
    "server": ("#e9e0f7", "#6b46c1"),
    "eval": ("#eceff1", "#455a64"),
    "xai": ("#fde2e2", "#c53030"),
}


def box(ax, x, y, w, h, text, kind, size=9, bold=False, align="center"):
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                facecolor=fill, edgecolor=edge, linewidth=1.5))
    tx = x + w / 2 if align == "center" else x + 0.8
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=size,
            fontweight="bold" if bold else "normal", color="#1a202c", linespacing=1.35)


def arrow(ax, p1, p2, color="#4a5568", both=False, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="<|-|>" if both else style, mutation_scale=13,
                                 color=color, linewidth=1.6, linestyle=ls))


def main() -> None:
    fig, ax = plt.subplots(figsize=(20, 12.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    ax.text(50, 64.3, "Heterogeneous Multimodal Federated Learning for Explainable, Resource-Aware TB Screening",
            ha="center", fontsize=17, fontweight="bold", color="#1a202c")
    ax.text(50, 62.2, "Architecture overview (single-machine simulation of 5 institutions; raw data never leaves a client)",
            ha="center", fontsize=11, color="#4a5568")

    # Institutions
    ax.text(12.5, 59.2, "1  INSTITUTIONS (local data + preprocessing)", ha="center", fontsize=10.5, fontweight="bold", color=COLORS["data"][1])
    clients = (
        "Hospital 1: Montgomery CXR\n138 images | grayscale to 3ch, 224x224,\nImageNet norm, flip/rotate aug",
        "Hospital 2: Shenzhen CXR\n662 images | same CXR preprocessing",
        "Hospital 3: Clinical split A\nDryad tabular records | 12 features\n(age, HIV, DM, contact, history...)",
        "Hospital 4: Clinical split B\nDryad tabular records | 12 features",
        "Hospital 5: CT (NIfTI)\n40 volumes (TB / NTM) | percentile\nwindowing, trilinear resize to 64^3",
    )
    bottoms = [51.8, 45.0, 38.2, 31.4, 24.6]
    for text, y in zip(clients, bottoms):
        box(ax, 1.5, y, 22, 6.0, text, "data", size=8.2)
    centers = [y + 3.0 for y in bottoms]

    # Encoders
    ax.text(35, 59.2, "2  LOCAL MODALITY ENCODERS", ha="center", fontsize=10.5, fontweight="bold", color=COLORS["enc"][1])
    cxr_y, cli_y, ct_y = 46.3, 33.4, 24.1
    box(ax, 27, cxr_y, 16, 10.6, "CXR encoder\nResNet18 (ImageNet)\nonly layer4 + projection\ntrainable\n\n-> 128-d embedding", "enc", size=8.6)
    box(ax, 27, cli_y, 16, 8.2, "Clinical encoder\nMLP 12 -> hidden -> 128\nDropout 0.2\n\n-> 128-d embedding", "enc", size=8.6)
    box(ax, 27, ct_y, 16, 6.4, "CT encoder\n3D CNN (16-32-64 ch)\n-> 128-d embedding", "enc", size=8.6)
    arrow(ax, (24.2, centers[0]), (26.8, cxr_y + 7.6))
    arrow(ax, (24.2, centers[1]), (26.8, cxr_y + 3.0))
    arrow(ax, (24.2, centers[2]), (26.8, cli_y + 6.0))
    arrow(ax, (24.2, centers[3]), (26.8, cli_y + 2.2))
    arrow(ax, (24.2, centers[4]), (26.8, ct_y + 3.2))

    # Fusion
    ax.text(62, 59.2, "3  MISSING-MODALITY-AWARE FUSION HEAD (shared)", ha="center", fontsize=10.5, fontweight="bold", color=COLORS["fusion"][1])
    ax.add_patch(FancyBboxPatch((47, 24.1), 30, 33.3, boxstyle="round,pad=0.3,rounding_size=1.0",
                                facecolor="#fffaf0", edgecolor=COLORS["fusion"][1], linewidth=2, linestyle="--"))
    box(ax, 48.5, 49.3, 27, 6.2, "Modality slots (cxr | clinical | ct):\nencoder embedding if present,\nlearned null embedding if absent", "fusion", size=8.6)
    box(ax, 48.5, 41.2, 27, 6.2, "Gate MLP on [embedding + presence flag]\n-> softmax weight per modality", "fusion", size=8.6)
    box(ax, 48.5, 33.1, 27, 6.2, "Weighted sum of embeddings\n-> fused 128-d patient vector", "fusion", size=8.6)
    box(ax, 48.5, 25.4, 27, 5.8, "Classifier MLP + Dropout 0.3\n(Dropout kept on for MC-Dropout)", "fusion", size=8.6)
    for y_top, y_bottom in ((49.3, 47.4), (41.2, 39.3), (33.1, 31.2)):
        arrow(ax, (62, y_top), (62, y_bottom))
    arrow(ax, (43.2, cxr_y + 5.3), (46.8, 53.8))
    arrow(ax, (43.2, cli_y + 4.1), (46.8, 52.4))
    arrow(ax, (43.2, ct_y + 3.2), (46.8, 51.0))

    # Output + trust
    ax.text(89, 59.2, "4  OUTPUT + TRUST LAYER", ha="center", fontsize=10.5, fontweight="bold", color=COLORS["xai"][1])
    box(ax, 81, 49.5, 17.5, 6.5, "TB probability\n(2-class softmax)\nthreshold tuned on\nvalidation bal. accuracy", "xai", size=8.4)
    box(ax, 81, 24.6, 17.5, 21.8,
        "Explainability\n- Grad-CAM heatmaps (CXR)\n\nUncertainty\n- MC-Dropout, 20 passes\n- refer least-certain cases\n  (uncertainty triage)\n\nInput checks\n- image-quality flags\n  (brightness, contrast,\n  sharpness, clipping)\n\nViews\n- clinician and patient\n  with non-diagnostic notice",
        "xai", size=8.2, align="left")
    arrow(ax, (77.3, 28.3), (80.8, 52.2))
    arrow(ax, (89.7, 49.3), (89.7, 46.6))

    # Server
    ax.text(40, 21.4, "5  FEDERATED SERVER (modality-aware aggregation, simulated on one machine)", ha="center", fontsize=10.5, fontweight="bold", color=COLORS["server"][1])
    box(ax, 1.5, 11.6, 22, 8.3, "Local training per client\nclass-weighted cross-entropy\nAdam, lr 3e-4, batch 16\nreturns updated weights", "server", size=8.6)
    box(ax, 27, 11.6, 33, 8.3, "Aggregation per round\nEncoders: averaged ONLY within the same modality\n(weighted by training samples)\nFusion head: averaged across ALL clients", "server", size=8.6)
    box(ax, 64, 11.6, 13.5, 8.3, "Broadcast\nupdated global\nmodel to clients", "server", size=8.6)
    arrow(ax, (23.7, 15.7), (26.8, 15.7), color=COLORS["server"][1])
    arrow(ax, (60.2, 15.7), (63.8, 15.7), color=COLORS["server"][1])
    arrow(ax, (12.5, 24.3), (12.5, 20.2), color=COLORS["server"][1], both=True)
    arrow(ax, (62, 23.8), (62, 20.2), color=COLORS["server"][1], both=True)
    ax.text(14.0, 22.2, "weights only,\nno raw data", fontsize=7.8, color=COLORS["server"][1], va="center")
    box(ax, 81, 11.6, 17.5, 8.3, "Repeated for all\nrounds (5-15)\nand 3 random seeds", "server", size=8.6)

    # Evaluation
    ax.text(50, 8.9, "6  EVALUATION PROTOCOL", ha="center", fontsize=10.5, fontweight="bold", color=COLORS["eval"][1])
    box(ax, 1.5, 1.2, 23.5, 6.8, "Splits\nstratified 70/15/15 per client\nthreshold chosen on validation only", "eval", size=8.4)
    box(ax, 26.5, 1.2, 28, 6.8, "Held-out test metrics\nsensitivity, specificity, balanced acc.,\nAUROC, AUPRC, F1, Brier, ECE, 95% bootstrap CI", "eval", size=8.4)
    box(ax, 56, 1.2, 21, 6.8, "Baselines\ncentralized (not private)\nunweighted FedAvg | local-only", "eval", size=8.4)
    box(ax, 78.5, 1.2, 20, 6.8, "Studies\nmodality ablation | blur/noise stress\nROC | selective prediction", "eval", size=8.4)

    plt.tight_layout()
    out = Path("ppt")
    out.mkdir(exist_ok=True)
    plt.savefig(out / "architecture.png", dpi=170, facecolor="white")
    print(out / "architecture.png")


if __name__ == "__main__":
    main()
