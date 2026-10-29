"""Phase 6: turns the saved training histories into a comparison table/plot of
macro accuracy across rounds for: our modality-aware FedAvg vs centralized vs
local-only vs plain (unweighted) FedAvg.

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

    series = {}
    fed_path = out_dir / "federated_history.json"
    if fed_path.exists():
        fed_history = load_json(fed_path)
        series["modality_aware_fedavg"] = [h["macro_accuracy"] for h in fed_history]

    base_path = out_dir / "baselines_history.json"
    if base_path.exists():
        baselines = load_json(base_path)
        for name, history in baselines.items():
            series[name] = [h["macro_accuracy"] for h in history]

    if not series:
        print("No history files found yet - run src.train_federated / src.train_baselines first.")
        return

    print("Final-round macro accuracy:")
    for name, values in series.items():
        print(f"  {name:28s} {values[-1]:.4f}  (best={max(values):.4f})")

    plt.figure(figsize=(8, 5))
    for name, values in series.items():
        plt.plot(range(1, len(values) + 1), values, marker="o", label=name)
    plt.xlabel("Round")
    plt.ylabel("Macro accuracy (mean across institutions)")
    plt.title("TB screening: federated vs centralized vs local-only vs plain FedAvg")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = out_dir / "comparison_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
