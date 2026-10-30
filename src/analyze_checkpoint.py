"""Uncertainty, ROC and Grad-CAM analysis of a saved federated checkpoint on the CXR clients.

Usage:
    python -m src.analyze_checkpoint --checkpoint checkpoints/study_seed42.pt --seed 42
Writes outputs/analysis.json and outputs/figures/{fig_roc,fig_selective,fig_gradcam}.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_curve

from src.data.clinical_dryad import FEATURE_COLUMNS
from src.data.cxr_dataset import _load_resized_image, build_transform
from src.explain.gradcam import CXRExplainer
from src.federated.client import binary_metrics, select_validation_threshold
from src.models.model_factory import build_encoder, build_fusion
from src.predict import mc_dropout_predict
from src.train_federated import CONFIG_PATH, build_cxr_client, set_seed

CLIENTS = (
    ("hospital_cxr_montgomery", "montgomery", "Montgomery (US)"),
    ("hospital_cxr_shenzhen", "shenzhen", "Shenzhen (China)"),
)
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def mc_predictions(encoder, fusion, loader, device, samples: int = 20):
    means, stds, labels = [], [], []
    for x, y in loader:
        mean, std = mc_dropout_predict(encoder, fusion, "cxr", x, device, n_samples=samples)
        means.extend(mean.cpu().tolist())
        stds.extend(std.cpu().tolist())
        labels.extend(y.tolist())
    return np.array(means), np.array(stds), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/study_seed42.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder = build_encoder("cxr", cfg["embedding_dim"], len(FEATURE_COLUMNS)).to(device)
    encoder.load_state_dict(checkpoint["encoders"]["cxr"])
    fusion = build_fusion(cfg["embedding_dim"]).to(device)
    fusion.load_state_dict(checkpoint["fusion"])

    fig_dir = Path(cfg["paths"]["outputs_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_client, pooled = {}, {"correct": [], "std": [], "label": [], "pred": []}
    clients = {}
    for name, source, _ in CLIENTS:
        client = build_cxr_client(
            name, cfg["paths"][f"{source}_dir"], source, cfg["image"]["cxr_size"],
            cfg["federated"]["batch_size"], args.seed,
        )
        clients[name] = client
        val_mean, _, val_labels = mc_predictions(encoder, fusion, client.val_loader, device)
        threshold = select_validation_threshold(val_labels.tolist(), val_mean.tolist())
        mean, std, labels = mc_predictions(encoder, fusion, client.test_loader, device)
        predictions = (mean >= threshold).astype(int)
        per_client[name] = {
            "threshold": threshold, "probs": mean, "std": std, "labels": labels,
            "metrics": binary_metrics(labels.tolist(), mean.tolist(), threshold),
        }
        pooled["std"].extend(std.tolist())
        pooled["label"].extend(labels.tolist())
        pooled["pred"].extend(predictions.tolist())

    std = np.array(pooled["std"])
    labels = np.array(pooled["label"])
    predictions = np.array(pooled["pred"])
    order = np.argsort(std)
    selective = []
    for coverage in COVERAGES:
        keep = order[: max(int(round(coverage * len(order))), 2)]
        kept_labels, kept_predictions = labels[keep], predictions[keep]
        entry = {"coverage": coverage, "n": int(len(keep)), "error_rate": float((kept_labels != kept_predictions).mean())}
        if len(set(kept_labels.tolist())) == 2:
            entry["balanced_accuracy"] = binary_metrics(
                kept_labels.tolist(), kept_predictions.astype(float).tolist(), 0.5
            )["balanced_accuracy"]
        selective.append(entry)

    # ROC curves
    plt.figure(figsize=(5.2, 4.6))
    for (name, _, title), color in zip(CLIENTS, ("#187a73", "#c0562a")):
        data = per_client[name]
        fpr, tpr, _ = roc_curve(data["labels"], data["probs"])
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{title}  AUROC={data['metrics']['auroc']:.2f} (n={len(data['labels'])})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Global federated model: held-out CXR ROC")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_roc.png", dpi=170)
    plt.close()

    # Selective prediction (uncertainty)
    plt.figure(figsize=(5.2, 4.2))
    plt.plot([100 * e["coverage"] for e in selective], [100 * e["error_rate"] for e in selective], marker="o", color="#527b9b", lw=2)
    plt.xlabel("Coverage: % of held-out CXRs auto-screened (lowest uncertainty first)")
    plt.ylabel("Error rate (%) on retained cases")
    plt.title("Uncertainty-aware triage (MC-Dropout)")
    plt.gca().invert_xaxis()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_selective.png", dpi=170)
    plt.close()

    # Grad-CAM gallery from the Shenzhen held-out set
    shenzhen = clients["hospital_cxr_shenzhen"]
    samples = shenzhen.test_loader.dataset.samples
    shenzhen_data = per_client["hospital_cxr_shenzhen"]
    chosen = []
    for target in (1, 1, 0, 0):
        for index, (_, label) in enumerate(samples):
            if label == target and index not in chosen:
                chosen.append(index)
                break
    explainer = CXRExplainer(encoder, fusion, device)
    transform = build_transform(cfg["image"]["cxr_size"], train=False)
    fig, axes = plt.subplots(2, len(chosen), figsize=(3.1 * len(chosen), 6.4))
    for column, index in enumerate(chosen):
        path, label = samples[index]
        resized = _load_resized_image(str(path), cfg["image"]["cxr_size"]).copy()
        rgb = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255
        tensor = transform(resized).unsqueeze(0).to(device)
        overlay = explainer.explain(tensor, rgb)
        probability = shenzhen_data["probs"][index]
        uncertainty = shenzhen_data["std"][index]
        axes[0, column].imshow(rgb)
        axes[0, column].set_title(f"{'TB' if label else 'Normal'} (reference)", fontsize=9)
        axes[1, column].imshow(overlay)
        axes[1, column].set_title(f"P(TB)={probability:.2f}  unc={uncertainty:.2f}", fontsize=9)
        for row in (0, 1):
            axes[row, column].axis("off")
    fig.suptitle("Grad-CAM evidence maps (clinician view; not a diagnosis)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_gradcam.png", dpi=150)
    plt.close()

    summary = {
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "selective_prediction": selective,
        "per_client": {k: {"threshold": v["threshold"], "metrics": v["metrics"]} for k, v in per_client.items()},
    }
    with open(Path(cfg["paths"]["outputs_dir"]) / "analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(selective, indent=2))


if __name__ == "__main__":
    main()
