"""Multi-seed study: modality-aware federation vs baselines, plus a modality-availability ablation.

Usage:
    python -m src.run_study --seeds 42 43 44 --rounds 5
Per-seed results are written to outputs/study/seed_<seed>.json and reused on rerun.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from src.data.clinical_dryad import FEATURE_COLUMNS
from src.federated.client import evaluate_heldout_client
from src.federated.server import FederatedServer
from src.train_baselines import (
    build_clients,
    mean_test_balanced_accuracy,
    run_centralized,
    run_local_only,
    run_plain_fedavg,
)
from src.train_federated import CONFIG_PATH, set_seed

ABLATION_SCENARIOS = {
    "cxr_only": ("cxr",),
    "cxr+clinical": ("cxr", "clinical"),
    "cxr+ct": ("cxr", "ct"),
}


def run_federated(clients, cfg, device, checkpoint_path: Path | None = None) -> dict:
    server = FederatedServer(
        clients=clients,
        embedding_dim=cfg["embedding_dim"],
        clinical_input_dim=len(FEATURE_COLUMNS),
        device=device,
        local_epochs=cfg["federated"]["local_epochs"],
        lr=cfg["federated"]["lr"],
    )
    curve = []
    for round_index in range(1, cfg["federated"]["num_rounds"] + 1):
        metrics = server.run_round()
        curve.append({"round": round_index, "macro_accuracy": metrics["macro_accuracy"]})
        print(f"    federated round {round_index}: val macro_acc={metrics['macro_accuracy']:.3f}", flush=True)

    test = {
        client.name: evaluate_heldout_client(
            client, server.global_encoders[client.modality], server.global_fusion,
            cfg["embedding_dim"], len(FEATURE_COLUMNS), device,
        )
        for client in clients
    }
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoders": server.global_encoders,
                "fusion": server.global_fusion,
                "metadata": {"seed": cfg["seed"], "rounds": cfg["federated"]["num_rounds"]},
            },
            checkpoint_path,
        )
    return {
        "curve": curve,
        "test_per_client": test,
        "test_mean_client_balanced_accuracy": mean_test_balanced_accuracy(test),
    }


def summarize_baseline(history: list[dict]) -> dict:
    return {
        "curve": [{"round": h["round"], "macro_accuracy": h["macro_accuracy"]} for h in history],
        "test_per_client": history[-1]["test_per_client"],
        "test_mean_client_balanced_accuracy": history[-1]["test_mean_client_balanced_accuracy"],
    }


def run_seed(seed: int, cfg: dict, device, out_dir: Path, checkpoint_dir: Path) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    result = {"seed": seed, "rounds": cfg["federated"]["num_rounds"], "methods": {}, "ablation": {}}

    print(f"[seed {seed}] modality-aware FedAvg", flush=True)
    set_seed(seed)
    checkpoint = checkpoint_dir / f"study_seed{seed}.pt"
    result["methods"]["modality_aware_fedavg"] = run_federated(build_clients(cfg), cfg, device, checkpoint)

    for name, runner in (
        ("centralized", run_centralized),
        ("plain_fedavg", run_plain_fedavg),
        ("local_only", run_local_only),
    ):
        print(f"[seed {seed}] {name}", flush=True)
        set_seed(seed)
        result["methods"][name] = summarize_baseline(runner(build_clients(cfg), cfg, device))

    for scenario, modalities in ABLATION_SCENARIOS.items():
        print(f"[seed {seed}] ablation {scenario}", flush=True)
        set_seed(seed)
        subset = [client for client in build_clients(cfg) if client.modality in modalities]
        result["ablation"][scenario] = run_federated(subset, cfg, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"seed_{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["federated"]["num_rounds"] = args.rounds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "study"
    checkpoint_dir = Path(cfg["paths"]["checkpoints_dir"])

    for seed in args.seeds:
        if (out_dir / f"seed_{seed}.json").exists():
            print(f"[seed {seed}] already complete, skipping", flush=True)
            continue
        run_seed(seed, cfg, device, out_dir, checkpoint_dir)
        print(f"[seed {seed}] done", flush=True)


if __name__ == "__main__":
    main()
