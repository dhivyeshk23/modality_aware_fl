"""Summarize final held-out test performance across training methods.

Usage:
    python -m src.evaluate_results
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["paths"]["outputs_dir"])

    comparisons = {}
    fed_test_path = out_dir / "federated_test_metrics.json"
    if fed_test_path.exists():
        metrics = load_json(fed_test_path)
        client_metrics = [
            value for value in metrics.values()
            if isinstance(value, dict) and "balanced_accuracy" in value
        ]
        comparisons["modality_aware_fedavg"] = summarize_test(client_metrics)

    base_path = out_dir / "baselines_history.json"
    if base_path.exists():
        baselines = load_json(base_path)
        for name, history in baselines.items():
            if history and "test_per_client" in history[-1]:
                comparisons[name] = summarize_test(list(history[-1]["test_per_client"].values()))

    if not comparisons:
        print("No held-out metrics found; run src.train_federated and src.train_baselines first.")
        return

    print("Mean client-level held-out test metrics:")
    for name, metrics in comparisons.items():
        print(f"  {name:28s} " + "  ".join(f"{key}={value:.3f}" for key, value in metrics.items()))

    with open(out_dir / "test_comparison.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    plt.figure(figsize=(8, 5))
    names = list(comparisons)
    values = [comparisons[name]["balanced_accuracy"] for name in names]
    bars = plt.bar(names, values, color=["#187a73", "#527b9b", "#9b7952", "#7b667e"][:len(names)])
    plt.bar_label(bars, fmt="%.3f", padding=3)
    plt.ylim(0, 1)
    plt.ylabel("Mean client balanced accuracy (held-out test)")
    plt.title("TB screening baselines: held-out test performance")
    plt.xticks(rotation=18, ha="right")
    plt.grid(alpha=0.3)
    out_path = out_dir / "comparison_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


def summarize_test(client_metrics: list[dict]) -> dict:
    metric_names = ("balanced_accuracy", "sensitivity", "specificity", "f1", "auroc", "auprc", "brier", "ece_10bin")
    summary = {}
    for name in metric_names:
        values = [m[name] for m in client_metrics if m.get(name) is not None]
        if values:
            summary[name] = sum(values) / len(values)
    return summary


if __name__ == "__main__":
    main()
