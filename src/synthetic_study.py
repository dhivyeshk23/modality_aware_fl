"""SYNTHETIC counterpart of the real study: 5 simulated institutions, 4 methods, ablation,
robustness, uncertainty triage, ROC, confusion matrices and a feature-attribution check.

All data are generated. Results describe this toy world only and are not TB performance.
Usage: python -m src.synthetic_study
Writes figures to ppt/synthetic/ and numbers to outputs/synthetic_study.json
"""
from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve

from src.federated.client import binary_metrics, select_validation_threshold
from src.models.fusion import MODALITIES, MissingModalityFusion

DIM, EMBED = 32, 32
REAL_DIM = {"cxr": 32, "clinical": 12, "ct": 32}
SIGNAL = {"cxr": 0.9, "clinical": 0.5, "ct": 0.7}
NOISE, PREVALENCE, SHIFT = 0.8, 0.4, 0.5
SEEDS = (0, 1, 2, 3, 4)
ROUNDS, LOCAL_EPOCHS = 20, 2
SPECS = (("cxr_a", "cxr", 120), ("cxr_b", "cxr", 80), ("clinical_a", "clinical", 100),
         ("clinical_b", "clinical", 60), ("ct", "ct", 60))
VAL_N, TEST_N = 60, 300
METHODS = (
    ("modality_aware", "Modality-aware FedAvg", "#187a73"),
    ("plain", "FedAvg, modality-unaware", "#527b9b"),
    ("local", "Local-only", "#c0562a"),
    ("centralized", "Centralized (not private)", "#8a8a8a"),
)
ABLATION = {"CXR only": ("cxr",), "CXR + clinical": ("cxr", "clinical"), "CXR + CT": ("cxr", "ct"),
            "All three": ("cxr", "clinical", "ct")}
OUT = Path("ppt/synthetic")
BANNER = "SYNTHETIC DATA - toy simulation, not TB performance"


def directions() -> dict[str, np.ndarray]:
    result = {}
    for index, modality in enumerate(MODALITIES):
        vector = np.random.default_rng(7 + index).normal(size=REAL_DIM[modality])
        result[modality] = np.pad(vector / np.linalg.norm(vector), (0, DIM - REAL_DIM[modality]))
    return result


DIRECTIONS = directions()


def generate(modality: str, n: int, rng: np.random.Generator, offset: np.ndarray):
    labels = (rng.random(n) < PREVALENCE).astype(int)
    mask = (np.arange(DIM) < REAL_DIM[modality]).astype(float)
    x = (2 * labels - 1)[:, None] * SIGNAL[modality] * DIRECTIONS[modality] + rng.normal(size=(n, DIM)) * NOISE * mask + offset
    return torch.tensor(x, dtype=torch.float32), torch.tensor(labels)


def make_clients(seed: int, modalities=MODALITIES) -> list[dict]:
    rng = np.random.default_rng(1000 + seed)
    clients = []
    for name, modality, n_train in SPECS:
        if modality not in modalities:
            continue
        mask = (np.arange(DIM) < REAL_DIM[modality]).astype(float)
        offset = rng.normal(size=DIM) * SHIFT * mask  # per-institution domain shift
        clients.append({
            "name": name, "modality": modality, "n": n_train,
            "train": generate(modality, n_train, rng, offset),
            "val": generate(modality, VAL_N, rng, offset),
            "test": generate(modality, TEST_N, rng, offset),
        })
    return clients


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({m: nn.Sequential(nn.Linear(DIM, 64), nn.ReLU(), nn.Linear(64, EMBED)) for m in MODALITIES})
        self.fusion = MissingModalityFusion(EMBED)

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        embedding = self.encoders[modality](x)
        return self.fusion({m: (embedding if m == modality else None) for m in MODALITIES}, x.size(0), "cpu")


def train_epochs(model: Net, batches: list, optimizer) -> None:
    model.train()
    for modality, x, y in batches:
        loss = nn.functional.cross_entropy(model(x, modality), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def client_batches(client: dict, epochs: int) -> list:
    x, y = client["train"]
    batches = []
    for _ in range(epochs):
        order = torch.randperm(len(y))
        batches += [(client["modality"], x[order[i:i + 32]], y[order[i:i + 32]]) for i in range(0, len(y), 32)]
    return batches


@torch.no_grad()
def predict(model: Net, modality: str, x: torch.Tensor, mc: int = 0):
    model.eval()
    if mc:
        for module in model.fusion.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    runs = [torch.softmax(model(x, modality), dim=1)[:, 1] for _ in range(max(mc, 1))]
    stacked = torch.stack(runs)
    return stacked.mean(0).numpy(), stacked.std(0, correction=min(len(runs) - 1, 1)).numpy()


def val_balanced_accuracy(models, clients) -> float:
    scores = []
    for model, client in zip(models, clients):
        probs, _ = predict(model, client["modality"], client["val"][0])
        scores.append(binary_metrics(client["val"][1].tolist(), probs.tolist(), 0.5)["balanced_accuracy"])
    return float(np.mean(scores))


def aggregate(models, clients, plain: bool) -> None:
    weights = np.array([c["n"] for c in clients], dtype=float)
    states = [m.state_dict() for m in models]
    for key in states[0]:
        if key.startswith("encoders.") and not plain:
            modality = key.split(".")[1]
            members = [i for i, c in enumerate(clients) if c["modality"] == modality]
        else:
            members = list(range(len(clients)))
        w = weights[members] / weights[members].sum()
        average = sum(states[i][key] * float(wi) for i, wi in zip(members, w))
        for i in members:
            states[i][key] = average.clone()
    for model, state in zip(models, states):
        model.load_state_dict(state)


def run_method(clients: list[dict], method: str, seed: int):
    torch.manual_seed(seed)
    curve = []
    if method == "centralized":
        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(ROUNDS):
            batches = [b for c in clients for b in client_batches(c, LOCAL_EPOCHS)]
            order = np.random.default_rng(seed).permutation(len(batches))
            train_epochs(model, [batches[i] for i in order], optimizer)
            curve.append(val_balanced_accuracy([model] * len(clients), clients))
        return [model] * len(clients), curve
    template = Net()
    models = [copy.deepcopy(template) for _ in clients]
    for _ in range(ROUNDS):
        for model, client in zip(models, clients):
            train_epochs(model, client_batches(client, LOCAL_EPOCHS), torch.optim.Adam(model.parameters(), lr=3e-3))
        if method != "local":
            aggregate(models, clients, plain=(method == "plain"))
        curve.append(val_balanced_accuracy(models, clients))
    return models, curve


def evaluate(models, clients, transform=None) -> dict:
    result = {}
    for model, client in zip(models, clients):
        val_probs, _ = predict(model, client["modality"], client["val"][0])
        threshold = select_validation_threshold(client["val"][1].tolist(), val_probs.tolist())
        x, y = client["test"]
        probs, _ = predict(model, client["modality"], transform(x, client["modality"]) if transform else x)
        metrics = binary_metrics(y.tolist(), probs.tolist(), threshold)
        preds = probs >= threshold
        labels = y.numpy()
        metrics.update(tp=int(((preds == 1) & (labels == 1)).sum()), tn=int(((preds == 0) & (labels == 0)).sum()),
                       fp=int(((preds == 1) & (labels == 0)).sum()), fn=int(((preds == 0) & (labels == 1)).sum()))
        result[client["name"]] = metrics
    return result


def noise_transform(level: float, seed: int):
    def apply(x, modality):
        mask = (torch.arange(DIM) < REAL_DIM[modality]).float()
        generator = torch.Generator().manual_seed(seed)
        return x + torch.randn(x.shape, generator=generator) * level * mask
    return apply


def mask_transform(fraction: float, seed: int):
    def apply(x, modality):
        generator = torch.Generator().manual_seed(seed)
        keep = (torch.rand(x.shape, generator=generator) >= fraction).float()
        return x * keep
    return apply


def mean_metric(per_client: dict, metric: str, names=None) -> float:
    names = names or list(per_client)
    return float(np.nanmean([np.nan if per_client[n][metric] is None else per_client[n][metric] for n in names]))


def titled(title: str) -> None:
    plt.title(f"{title}\n{BANNER}", color="#b00020", fontsize=9, fontweight="bold")


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=170)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_names = [s[0] for s in SPECS]
    cxr_names = ["cxr_a", "cxr_b"]
    main_runs, ablation, robustness, selective = {}, {}, {"noise": {}, "mask": {}}, []
    roc_data, attribution = {}, None

    for seed in SEEDS:
        clients = make_clients(seed)
        main_runs[seed] = {}
        for method, _, _ in METHODS:
            models, curve = run_method(clients, method, seed)
            main_runs[seed][method] = {"per_client": evaluate(models, clients), "curve": curve}
            if method == "modality_aware":
                for level in (0.0, 0.25, 0.5, 1.0):
                    result = evaluate(models, clients, noise_transform(level, seed) if level else None)
                    robustness["noise"].setdefault(level, []).append(mean_metric(result, "balanced_accuracy", cxr_names))
                for fraction in (0.0, 0.25, 0.5, 0.75):
                    result = evaluate(models, clients, mask_transform(fraction, seed) if fraction else None)
                    robustness["mask"].setdefault(fraction, []).append(mean_metric(result, "balanced_accuracy", cxr_names))

                stds, errors = [], []
                for model, client in zip(models, clients):
                    if client["name"] not in cxr_names:
                        continue
                    val_probs, _ = predict(model, "cxr", client["val"][0])
                    threshold = select_validation_threshold(client["val"][1].tolist(), val_probs.tolist())
                    probs, std = predict(model, "cxr", client["test"][0], mc=20)
                    labels = client["test"][1].numpy()
                    errors += list((probs >= threshold) != labels)
                    stds += list(std)
                    if seed == SEEDS[0]:
                        roc_data[client["name"]] = (labels, probs, main_runs[seed][method]["per_client"][client["name"]]["auroc"])
                order = np.argsort(stds)
                errors = np.array(errors)[order]
                selective.append([float(errors[: int(round(c * len(errors)))].mean()) for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)])

                if seed == SEEDS[0]:
                    client = clients[0]
                    x = client["test"][0][client["test"][1] == 1].clone().requires_grad_(True)
                    model = models[0]
                    model.eval()
                    torch.softmax(model(x, "cxr"), dim=1)[:, 1].sum().backward()
                    attribution = (x.grad * x).abs().mean(0).detach().numpy()

        ablation[seed] = {}
        for label, modalities in ABLATION.items():
            subset = make_clients(seed, modalities)
            models, _ = run_method(subset, "modality_aware", seed)
            ablation[seed][label] = mean_metric(evaluate(models, subset), "balanced_accuracy", cxr_names)
        print(f"seed {seed} done", flush=True)

    summary = {"note": BANNER}

    # metrics table
    metric_keys = (("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("balanced_accuracy", "Balanced acc."),
                   ("auroc", "AUROC"), ("auprc", "AUPRC"), ("f1", "F1"))
    rows = []
    for method, label, _ in METHODS:
        row = [label]
        for key, _ in metric_keys:
            values = [mean_metric(main_runs[s][method]["per_client"], key) for s in SEEDS]
            row.append(f"{np.mean(values):.3f} ± {np.std(values):.3f}")
            summary.setdefault(method, {})[key] = (float(np.mean(values)), float(np.std(values)))
        rows.append(row)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Method"] + [m[1] for m in metric_keys], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    table.auto_set_column_width(list(range(len(metric_keys) + 1)))
    for column in range(len(metric_keys) + 1):
        table[0, column].set_facecolor("#1f4a7c")
        table[0, column].set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Held-out test metrics, mean of 5 clients ({len(SEEDS)} seeds)\n{BANNER}", fontsize=10, color="#b00020")
    save("metrics_table.png")

    # confusion matrices (seed 0, main method)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4))
    for ax, name in zip(axes, all_names):
        d = main_runs[SEEDS[0]]["modality_aware"]["per_client"][name]
        matrix = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13, color="white" if value > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred neg", "Pred pos"], fontsize=8)
        ax.set_yticks([0, 1], ["True neg", "True pos"], fontsize=8)
        ax.set_title(f"{name} (n={d['n']})", fontsize=9)
    fig.suptitle(f"Confusion matrices, modality-aware FedAvg, seed {SEEDS[0]}\n{BANNER}", fontsize=10, color="#b00020")
    save("confusion_matrices.png")

    # fig1 method comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, title in ((axes[0], "balanced_accuracy", "Mean client balanced accuracy"), (axes[1], "auroc", "Mean client AUROC")):
        for index, (method, label, color) in enumerate(METHODS):
            values = [mean_metric(main_runs[s][method]["per_client"], key) for s in SEEDS]
            ax.bar(index, np.mean(values), yerr=np.std(values), capsize=4, color=color)
            ax.scatter([index] * len(values), values, color="black", s=14, zorder=3)
            ax.text(index, np.mean(values) + np.std(values) + 0.01, f"{np.mean(values):.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(4), [textwrap.fill(m[1], 14) for m in METHODS], fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Method comparison ({len(SEEDS)} seeds)\n{BANNER}", color="#b00020", fontsize=10, fontweight="bold")
    save("fig1_method_comparison.png")

    # fig2 per client
    plt.figure(figsize=(11, 4.4))
    for offset, (method, label, color) in enumerate(METHODS):
        means = [np.mean([main_runs[s][method]["per_client"][n]["balanced_accuracy"] for s in SEEDS]) for n in all_names]
        stds = [np.std([main_runs[s][method]["per_client"][n]["balanced_accuracy"] for s in SEEDS]) for n in all_names]
        plt.bar(np.arange(5) + (offset - 1.5) * 0.2, means, 0.2, yerr=stds, capsize=2, label=label, color=color)
    plt.xticks(range(5), all_names)
    plt.ylabel("Held-out balanced accuracy")
    plt.ylim(0.4, 1.0)
    plt.legend(fontsize=8, ncol=2)
    plt.grid(axis="y", alpha=0.3)
    titled("Per-client balanced accuracy (mean ± std over seeds)")
    save("fig2_per_client.png")

    # fig3 ablation
    plt.figure(figsize=(6.5, 4.4))
    for index, label in enumerate(ABLATION):
        values = [ablation[s][label] for s in SEEDS]
        plt.bar(index, np.mean(values), yerr=np.std(values), capsize=4, color=["#8a8a8a", "#527b9b", "#c0562a", "#187a73"][index])
        plt.scatter([index] * len(values), values, color="black", s=14, zorder=3)
        plt.text(index, np.mean(values) + np.std(values) + 0.01, f"{np.mean(values):.3f}", ha="center", fontsize=8)
        summary.setdefault("ablation", {})[label] = (float(np.mean(values)), float(np.std(values)))
    plt.xticks(range(4), list(ABLATION), fontsize=9)
    plt.ylim(0.4, 1.0)
    plt.ylabel("CXR clients' balanced accuracy")
    plt.grid(axis="y", alpha=0.3)
    titled("Which participating modalities help the CXR clients?")
    save("fig3_modality_ablation.png")

    # fig4 robustness
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, xlabel in ((axes[0], "noise", "Added feature noise (std)"), (axes[1], "mask", "Fraction of features masked")):
        xs = sorted(robustness[key])
        means = np.array([np.mean(robustness[key][x]) for x in xs])
        stds = np.array([np.std(robustness[key][x]) for x in xs])
        ax.plot(xs, means, marker="o", color="#187a73")
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color="#187a73")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CXR clients' balanced accuracy")
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.3)
        summary.setdefault("robustness", {})[key] = {str(x): float(np.mean(robustness[key][x])) for x in xs}
    fig.suptitle(f"Input-degradation stress test (analogue of blur/noise)\n{BANNER}", color="#b00020", fontsize=10, fontweight="bold")
    save("fig4_robustness.png")

    # fig5 convergence
    plt.figure(figsize=(6.2, 4.2))
    for method, label, color in METHODS:
        curves = np.array([main_runs[s][method]["curve"] for s in SEEDS])
        rounds = np.arange(1, curves.shape[1] + 1)
        plt.plot(rounds, curves.mean(0), color=color, label=label)
        plt.fill_between(rounds, curves.mean(0) - curves.std(0), curves.mean(0) + curves.std(0), color=color, alpha=0.12)
    plt.xlabel("Round")
    plt.ylabel("Validation balanced accuracy")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    titled("Convergence (mean ± std over seeds)")
    save("fig5_convergence.png")

    # ROC
    plt.figure(figsize=(5.2, 4.6))
    for (name, (labels, probs, auroc)), color in zip(roc_data.items(), ("#187a73", "#c0562a")):
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} AUROC={auroc:.2f} (n={len(labels)})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    titled("Held-out ROC, modality-aware FedAvg")
    save("fig_roc.png")

    # selective prediction
    coverage = (100, 90, 80, 70, 60, 50)
    errors = 100 * np.array(selective)
    plt.figure(figsize=(5.2, 4.2))
    plt.errorbar(coverage, errors.mean(0), yerr=errors.std(0), marker="o", color="#527b9b", capsize=3)
    plt.gca().invert_xaxis()
    plt.xlabel("Coverage: % auto-screened (lowest uncertainty first)")
    plt.ylabel("Error rate (%) on retained cases")
    plt.grid(alpha=0.3)
    titled("Uncertainty-aware triage (MC-Dropout, mean ± std over seeds)")
    save("fig_selective.png")
    summary["selective_error_pct"] = dict(zip(map(str, coverage), map(float, errors.mean(0))))

    # attribution vs known ground truth (the Grad-CAM analogue)
    truth = np.abs(DIRECTIONS["cxr"])
    correlation = float(np.corrcoef(truth, attribution)[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, data, title in ((axes[0], truth, "True signal direction |d|"), (axes[1], attribution, f"Model attribution (gradient x input)")):
        ax.imshow(data.reshape(4, 8), cmap="inferno")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Attribution check, Pearson r = {correlation:.2f}\n{BANNER}", color="#b00020", fontsize=10, fontweight="bold")
    save("fig_attribution.png")
    summary["attribution_correlation"] = correlation

    with open("outputs/synthetic_study.json", "w") as f:
        json.dump(summary, f, indent=2)
    for method, label, _ in METHODS:
        print(label, {k: round(v[0], 3) for k, v in summary[method].items()})
    print("ablation", {k: round(v[0], 3) for k, v in summary["ablation"].items()})
    print("robustness", summary["robustness"])
    print("selective", {k: round(v, 1) for k, v in summary["selective_error_pct"].items()})
    print("attribution r", round(correlation, 3))


if __name__ == "__main__":
    main()
