"""Re-evaluate the exact saved global checkpoint on held-out client test sets.

Usage:
    python -m src.evaluate_checkpoint
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.data.clinical_dryad import FEATURE_COLUMNS
from src.federated.client import evaluate_heldout_client
from src.train_federated import (
    CONFIG_PATH,
    build_clinical_clients,
    build_ct_client,
    build_cxr_client,
    set_seed,
)


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(cfg["paths"]["checkpoints_dir"]) / "federated_global.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    if not metadata:
        history_path = Path(cfg["paths"]["outputs_dir"]) / "federated_history.json"
        if history_path.exists():
            with open(history_path) as f:
                metadata = {"seed": seed, "rounds": len(json.load(f)), "source": "history_fallback"}
    if metadata and metadata.get("seed") != seed:
        raise ValueError(f"checkpoint seed {metadata.get('seed')} does not match config seed {seed}")

    batch_size = cfg["federated"]["batch_size"]
    clients = [
        build_cxr_client(
            "hospital_cxr_montgomery", cfg["paths"]["montgomery_dir"], "montgomery",
            cfg["image"]["cxr_size"], batch_size, seed,
        ),
        build_cxr_client(
            "hospital_cxr_shenzhen", cfg["paths"]["shenzhen_dir"], "shenzhen",
            cfg["image"]["cxr_size"], batch_size, seed,
        ),
        *build_clinical_clients(cfg["paths"]["dryad_xlsx"], 2, batch_size, seed),
        build_ct_client(
            "hospital_ct", cfg["paths"]["ct_subset_dir"], cfg["image"]["ct_size"],
            batch_size=4, seed=seed,
        ),
    ]

    metrics = {
        client.name: evaluate_heldout_client(
            client,
            checkpoint["encoders"][client.modality],
            checkpoint["fusion"],
            cfg["embedding_dim"],
            len(FEATURE_COLUMNS),
            device,
        )
        for client in clients
    }
    metrics["mean_client_balanced_accuracy"] = sum(
        metrics[client.name]["balanced_accuracy"] for client in clients
    ) / len(clients)
    metrics["checkpoint_metadata"] = metadata
    metrics["checkpoint_path"] = str(checkpoint_path)

    out_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "federated_test_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Rounds: {metadata.get('rounds', 'unknown')}; seed: {metadata.get('seed', seed)}")
    print(f"Held-out mean client balanced accuracy: {metrics['mean_client_balanced_accuracy']:.4f}")
    print(f"Saved exact-checkpoint metrics to {output_path}")


if __name__ == "__main__":
    main()