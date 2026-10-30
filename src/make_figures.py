"""Build comparison figures and summary tables from outputs/study/seed_*.json.

Usage: python -m src.make_figures
Writes outputs/figures/fig1..fig5 and outputs/study_summary.{csv,json}.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs")
FIG = OUT / "figures"
METHODS = (
    ("modality_aware_fedavg", "Modality-aware FedAvg (ours)", "#187a73"),
    ("plain_fedavg", "FedAvg, unweighted", "#527b9b"),
    ("local_only", "Local-only", "#c0562a"),
    ("centralized", "Centralized (not private)", "#8a8a8a"),
)
CLIENTS = (
    ("hospital_cxr_montgomery", "CXR\nMontgomery"),
    ("hospital_cxr_shenzhen", "CXR\nShenzhen"),
    ("hospital_clinical_a", "Clinical\nsplit A"),
    ("hospital_clinical_b", "Clinical\nsplit B"),
    ("hospital_ct", "CT"),
)
CXR_CLIENTS = ("hospital_cxr_montgomery", "hospital_cxr_shenzhen")


def load_seeds() -> list[dict]:
    files = sorted((OUT / "study").glob("seed_*.json"))
    if not files:
        raise SystemExit("no study results found; run src.run_study first")
    return [json.load(open(f)) for f in files]


def client_metric(run: dict, client: str, metric: str) -> float:
    value = run["test_per_client"][client].get(metric)
    return float("nan") if value is None else float(value)


def mean_over(run: dict, clients, metric: str) -> float:
    return float(np.nanmean([client_metric(run, c, metric) for c in clients]))


def annotate_bars(ax, bars, values, stds):
    for bar, value, std in zip(bars, values, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, value + std + 0.01, f"{value:.3f}", ha="center", fontsize=8)


def fig_methods(seeds, summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in (
        (axes[0], "balanced_accuracy", "Mean client balanced accuracy"),
        (axes[1], "auroc", "Mean client AUROC"),
    ):
        all_clients = [c for c, _ in CLIENTS]
        values = []
        for key, label, color in METHODS:
            per_seed = [mean_over(s["methods"][key], all_clients, metric) for s in seeds]
            values.append((label, color, per_seed))
            summary.setdefault(key, {})[f"mean_{metric}"] = float(np.mean(per_seed))
            summary[key][f"std_{metric}"] = float(np.std(per_seed))
        means = [np.mean(v[2]) for v in values]
        stds = [np.std(v[2]) for v in values]
        bars = ax.bar(range(len(values)), means, yerr=stds, capsize=4, color=[v[1] for v in values], alpha=0.9)
        for index, (_, _, per_seed) in enumerate(values):
            ax.scatter([index] * len(per_seed), per_seed, color="black", s=14, zorder=3)
        annotate_bars(ax, bars, means, stds)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([v[0].replace(" (", "\n(").replace(", ", ",\n").replace("FedAvg ", "FedAvg\n") for v in values], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Held-out comparison, mean of 5 clients (n={len(seeds)} seeds; dots = seeds)")
    plt.tight_layout()
    plt.savefig(FIG / "fig1_method_comparison.png", dpi=170)
    plt.close()


def fig_per_client(seeds, summary):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    width = 0.2
    for offset, (key, label, color) in enumerate(METHODS):
        means, stds = [], []
        for client, _ in CLIENTS:
            per_seed = [client_metric(s["methods"][key], client, "balanced_accuracy") for s in seeds]
            means.append(np.mean(per_seed))
            stds.append(np.std(per_seed))
            summary.setdefault("per_client", {}).setdefault(client, {})[key] = {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed))}
        ax.bar(np.arange(len(CLIENTS)) + (offset - 1.5) * width, means, width, yerr=stds, capsize=2, label=label, color=color)
    ax.set_xticks(range(len(CLIENTS)))
    ax.set_xticklabels([c[1] for c in CLIENTS])
    ax.set_ylabel("Held-out balanced accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Per-client balanced accuracy (mean ± std over seeds)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig2_per_client.png", dpi=170)
    plt.close()


def fig_ablation(seeds, summary):
    labels = ["CXR only\n(no other\nmodality)", "CXR +\nclinical", "CXR + CT", "All three\nmodalities"]
    sources = ("cxr_only", "cxr+clinical", "cxr+ct", None)
    means, stds, per_seed_all = [], [], []
    for source in sources:
        per_seed = []
        for s in seeds:
            run = s["methods"]["modality_aware_fedavg"] if source is None else s["ablation"][source]
            per_seed.append(mean_over(run, CXR_CLIENTS, "balanced_accuracy"))
        means.append(np.mean(per_seed))
        stds.append(np.std(per_seed))
        per_seed_all.append(per_seed)
    summary["ablation_cxr_balanced_accuracy"] = {
        label.replace("\n", " "): {"mean": float(m), "std": float(sd)} for label, m, sd in zip(labels, means, stds)
    }
    plt.figure(figsize=(6.5, 4.4))
    bars = plt.bar(range(4), means, yerr=stds, capsize=4, color=["#8a8a8a", "#527b9b", "#c0562a", "#187a73"])
    for index, per_seed in enumerate(per_seed_all):
        plt.scatter([index] * len(per_seed), per_seed, color="black", s=14, zorder=3)
    annotate_bars(plt.gca(), bars, means, stds)
    plt.xticks(range(4), labels, fontsize=9)
    plt.ylim(0.5, 1.0)
    plt.ylabel("CXR clients' held-out balanced accuracy")
    plt.title("Which participating modalities help the CXR models?")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig3_modality_ablation.png", dpi=170)
    plt.close()


def fig_convergence(seeds):
    plt.figure(figsize=(6.2, 4.2))
    for key, label, color in METHODS:
        curves = np.array([[point["macro_accuracy"] for point in s["methods"][key]["curve"]] for s in seeds])
        rounds = np.arange(1, curves.shape[1] + 1)
        plt.plot(rounds, curves.mean(axis=0), marker="o", color=color, label=label)
        plt.fill_between(rounds, curves.mean(axis=0) - curves.std(axis=0), curves.mean(axis=0) + curves.std(axis=0), color=color, alpha=0.12)
    plt.xlabel("Training round / epoch")
    plt.ylabel("Validation macro accuracy")
    plt.title("Convergence (validation, mean ± std over seeds)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig5_convergence.png", dpi=170)
    plt.close()


def fig_robustness():
    path = OUT / "cxr_quality_stress.json"
    if not path.exists():
        return
    data = json.load(open(path))
    groups = (
        ("Gaussian blur (sigma, px)", ["clean", "blur_1", "blur_2", "blur_4"], [0, 1, 2, 4]),
        ("Gaussian noise (std, normalized units)", ["clean", "noise_0.05", "noise_0.15", "noise_0.30"], [0, 0.05, 0.15, 0.30]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, keys, xs) in zip(axes, groups):
        for (client, label), color in zip(CLIENTS[:2], ("#187a73", "#c0562a")):
            if client not in data:
                continue
            ys = [data[client][k]["balanced_accuracy"] for k in keys]
            ax.plot(xs, ys, marker="o", color=color, label=label.replace("\n", " "))
        ax.set_xlabel(title)
        ax.set_ylabel("Held-out balanced accuracy")
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Image-quality stress test of the global federated model")
    plt.tight_layout()
    plt.savefig(FIG / "fig4_robustness.png", dpi=170)
    plt.close()


def write_tables(seeds, summary):
    rows = []
    for key, label, _ in METHODS:
        entry = summary[key]
        rows.append({
            "method": label,
            "balanced_accuracy_mean": round(entry["mean_balanced_accuracy"], 4),
            "balanced_accuracy_std": round(entry["std_balanced_accuracy"], 4),
            "auroc_mean": round(entry["mean_auroc"], 4),
            "auroc_std": round(entry["std_auroc"], 4),
        })
    with open(OUT / "study_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["seeds"] = [s["seed"] for s in seeds]
    summary["rounds"] = seeds[0]["rounds"]
    with open(OUT / "study_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for row in rows:
        print(row)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    seeds = load_seeds()
    summary: dict = {}
    fig_methods(seeds, summary)
    fig_per_client(seeds, summary)
    fig_ablation(seeds, summary)
    fig_convergence(seeds)
    fig_robustness()
    write_tables(seeds, summary)
    print("ablation:", json.dumps(summary["ablation_cxr_balanced_accuracy"], indent=1))
    print("per-client:", json.dumps(summary["per_client"], indent=1))
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
