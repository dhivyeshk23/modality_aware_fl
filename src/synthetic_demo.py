"""SYNTHETIC mechanism demo for paired multimodal fusion with missing modalities.

Patients are generated, not real. Each modality carries a noisy signal of a latent label, so
"more modalities help" is true by construction. This checks that MissingModalityFusion
trains and degrades gracefully when modalities are absent; it says nothing about TB accuracy.

Usage: python -m src.synthetic_demo
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.models.fusion import MODALITIES, MissingModalityFusion

DIMS = {"cxr": 32, "clinical": 12, "ct": 32}
SIGNAL = {"cxr": 0.9, "clinical": 0.5, "ct": 0.7}  # per-modality separation between classes
EMBED = 32
SEEDS = (0, 1, 2, 3, 4)


def make_patients(n: int, rng: np.random.Generator, noise: float = 0.6) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    labels = rng.integers(0, 2, size=n)
    sign = (2 * labels - 1)[:, None]
    data = {}
    for modality, dim in DIMS.items():
        direction = np.random.default_rng(DIMS[modality]).normal(size=dim)
        direction /= np.linalg.norm(direction)
        data[modality] = torch.tensor(
            sign * SIGNAL[modality] * direction + rng.normal(size=(n, dim)) * noise, dtype=torch.float32
        ).float()
    return data, torch.tensor(labels)


class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {m: nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, EMBED)) for m, d in DIMS.items()}
        )
        self.fusion = MissingModalityFusion(EMBED)

    def forward(self, data: dict[str, torch.Tensor], present: tuple[str, ...]) -> torch.Tensor:
        n = next(iter(data.values())).size(0)
        embeddings = {m: (self.encoders[m](data[m]) if m in present else None) for m in MODALITIES}
        return self.fusion(embeddings, batch_size=n, device="cpu")


def train(data, labels, modality_dropout: bool, seed: int) -> DemoModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DemoModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    subsets = [s for r in (1, 2, 3) for s in itertools.combinations(MODALITIES, r)]
    for _ in range(150):
        model.train()
        order = torch.randperm(len(labels))
        for start in range(0, len(labels), 64):
            index = order[start:start + 64]
            batch = {m: data[m][index] for m in MODALITIES}
            present = subsets[rng.integers(len(subsets))] if modality_dropout else MODALITIES
            loss = nn.functional.cross_entropy(model(batch, present), labels[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(model: DemoModel, data, labels, present) -> dict:
    model.eval()
    probs = torch.softmax(model(data, present), dim=1)[:, 1].numpy()
    return {
        "auroc": float(roc_auc_score(labels.numpy(), probs)),
        "balanced_accuracy": float(balanced_accuracy_score(labels.numpy(), probs >= 0.5)),
    }


def main() -> None:
    subsets = [s for r in (1, 2, 3) for s in itertools.combinations(MODALITIES, r)]
    names = ["+".join(s) for s in subsets]
    results = {"all_modalities_only": {n: [] for n in names}, "with_modality_dropout": {n: [] for n in names}}
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        train_data, train_labels = make_patients(600, rng)
        test_data, test_labels = make_patients(1000, rng)
        for key, dropout in (("all_modalities_only", False), ("with_modality_dropout", True)):
            model = train(train_data, train_labels, dropout, seed)
            for subset, name in zip(subsets, names):
                results[key][name].append(evaluate(model, test_data, test_labels, subset)["auroc"])

    summary = {k: {n: (float(np.mean(v)), float(np.std(v))) for n, v in d.items()} for k, d in results.items()}
    out = Path("outputs")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    with open(out / "synthetic_demo.json", "w") as f:
        json.dump({"note": "SYNTHETIC DATA - mechanism demo only", "auroc_mean_std": summary}, f, indent=2)

    x = np.arange(len(names))
    plt.figure(figsize=(10, 4.6))
    for offset, (key, label, color) in enumerate((
        ("all_modalities_only", "Trained on all modalities only", "#c0562a"),
        ("with_modality_dropout", "Trained with random modality dropout", "#187a73"),
    )):
        means = [summary[key][n][0] for n in names]
        stds = [summary[key][n][1] for n in names]
        plt.bar(x + (offset - 0.5) * 0.38, means, 0.38, yerr=stds, capsize=2, label=label, color=color)
    plt.xticks(x, [n.replace("+", "\n+") for n in names], fontsize=8)
    plt.ylabel("AUROC (synthetic patients)")
    plt.ylim(0.4, 1.0)
    plt.xlabel("Modalities available at test time")
    plt.title("SYNTHETIC DATA - fusion mechanism check, not TB performance", color="#b00020", fontweight="bold")
    plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "figures" / "synthetic_fusion_demo.png", dpi=170)
    for key in summary:
        print(key, {n: round(v[0], 3) for n, v in summary[key].items()})


if __name__ == "__main__":
    main()
