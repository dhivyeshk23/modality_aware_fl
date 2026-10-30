"""SYNTHETIC experiments for learning (not real TB data): noise sweep, training-size sweep,
and a small federated simulation with unpaired single-modality clients.

Signals are generated so every modality is informative; conclusions apply to this toy setup only.
Usage: python -m src.synthetic_experiments
Writes outputs/figures/synthetic_{noise,size,federated}.png and outputs/synthetic_experiments.json
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.models.fusion import MODALITIES
from src.synthetic_demo import DemoModel, evaluate, make_patients, train

OUT = Path("outputs")
SEEDS = (0, 1, 2)
BANNER = "SYNTHETIC DATA - toy experiment, not TB performance"


def banner(title: str) -> None:
    plt.title(f"{title}\n{BANNER}", color="#b00020", fontsize=9, fontweight="bold")


def noise_sweep() -> dict:
    levels = (0.3, 0.6, 0.9, 1.2, 1.5, 2.0)
    results = {name: {n: [] for n in levels} for name in ("clinical", "cxr", "cxr+clinical+ct")}
    for seed in SEEDS:
        for noise in levels:
            rng = np.random.default_rng(200 + seed)
            train_data, train_labels = make_patients(600, rng, noise)
            test_data, test_labels = make_patients(1000, rng, noise)
            model = train(train_data, train_labels, True, seed)
            for name in results:
                results[name][noise].append(evaluate(model, test_data, test_labels, tuple(name.split("+")))["auroc"])
    return {n: {str(k): (float(np.mean(v)), float(np.std(v))) for k, v in d.items()} for n, d in results.items()}


def size_sweep() -> dict:
    sizes = (40, 80, 160, 320, 640)
    results = {name: {n: [] for n in sizes} for name in ("clinical", "cxr", "cxr+clinical+ct")}
    for seed in SEEDS:
        for size in sizes:
            rng = np.random.default_rng(300 + seed)
            train_data, train_labels = make_patients(size, rng)
            test_data, test_labels = make_patients(1000, rng)
            model = train(train_data, train_labels, True, seed)
            for name in results:
                results[name][size].append(evaluate(model, test_data, test_labels, tuple(name.split("+")))["auroc"])
    return {n: {str(k): (float(np.mean(v)), float(np.std(v))) for k, v in d.items()} for n, d in results.items()}


def local_epochs(model: DemoModel, modality: str, data, labels, epochs: int, optimizer) -> None:
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(labels))
        for start in range(0, len(labels), 32):
            index = order[start:start + 32]
            batch = {m: data[m][index] for m in MODALITIES}
            loss = nn.functional.cross_entropy(model(batch, (modality,)), labels[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def average(states: list[dict], weights: list[float]) -> dict:
    total = sum(weights)
    return {k: sum(s[k] * w for s, w in zip(states, weights)) / total for k in states[0]}


def federated_run(clients: list[dict], rounds: int, epochs: int, seed: int, share: bool) -> DemoModel:
    """share=True: modality-aware FedAvg (encoders per modality, fusion head across all). share=False: local only."""
    torch.manual_seed(seed)
    template = DemoModel()
    models = [copy.deepcopy(template) for _ in clients]
    for _ in range(rounds):
        for client, model in zip(clients, models):
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            local_epochs(model, client["modality"], client["data"], client["labels"], epochs, optimizer)
        if not share:
            continue
        weights = [len(c["labels"]) for c in clients]
        fusion = average([m.fusion.state_dict() for m in models], weights)
        for modality in MODALITIES:
            members = [i for i, c in enumerate(clients) if c["modality"] == modality]
            encoder = average([models[i].encoders[modality].state_dict() for i in members], [weights[i] for i in members])
            for i in members:
                models[i].encoders[modality].load_state_dict(encoder)
        for model in models:
            model.fusion.load_state_dict(fusion)
    return models


def federated_sweep() -> dict:
    sizes = (20, 40, 80, 160)
    methods = ("Local-only", "Modality-aware FedAvg")
    results = {m: {n: [] for n in sizes} for m in methods}
    for seed in SEEDS:
        for size in sizes:
            rng = np.random.default_rng(400 + seed)
            clients = []
            for modality in MODALITIES:
                for _ in range(2):  # two institutions per modality
                    data, labels = make_patients(size, rng)
                    clients.append({"modality": modality, "data": data, "labels": labels})
            test_data, test_labels = make_patients(1000, rng)
            for method, share in zip(methods, (False, True)):
                models = federated_run(clients, rounds=15, epochs=2, seed=seed, share=share)
                aurocs = [evaluate(m, test_data, test_labels, (c["modality"],))["auroc"] for m, c in zip(models, clients)]
                results[method][size].append(float(np.mean(aurocs)))
    return {m: {str(k): (float(np.mean(v)), float(np.std(v))) for k, v in d.items()} for m, d in results.items()}


def line_plot(data: dict, xlabel: str, title: str, path: Path, colors: dict | None = None) -> None:
    plt.figure(figsize=(6.2, 4.4))
    for name, points in data.items():
        xs = [float(k) for k in points]
        means = np.array([v[0] for v in points.values()])
        stds = np.array([v[1] for v in points.values()])
        plt.plot(xs, means, marker="o", label=name, color=(colors or {}).get(name))
        plt.fill_between(xs, means - stds, means + stds, alpha=0.15, color=(colors or {}).get(name))
    plt.xlabel(xlabel)
    plt.ylabel("AUROC (synthetic test patients)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    banner(title)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    noise = noise_sweep()
    size = size_sweep()
    federated = federated_sweep()
    line_plot(noise, "Feature noise level", "Fusion vs single modality as noise grows", OUT / "figures" / "synthetic_noise.png")
    line_plot(size, "Training patients (paired)", "Learning curve", OUT / "figures" / "synthetic_size.png")
    line_plot(federated, "Patients per institution (6 institutions, one modality each)",
              "Federated vs local-only, small institutions", OUT / "figures" / "synthetic_federated.png",
              {"Local-only": "#c0562a", "Modality-aware FedAvg": "#187a73"})
    with open(OUT / "synthetic_experiments.json", "w") as f:
        json.dump({"note": BANNER, "noise": noise, "size": size, "federated": federated}, f, indent=2)
    for name, table in (("noise", noise), ("size", size), ("federated", federated)):
        print(name, {k: {x: round(v[0], 3) for x, v in d.items()} for k, d in table.items()})


if __name__ == "__main__":
    main()
