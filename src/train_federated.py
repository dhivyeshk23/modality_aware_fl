"""Phase 3 entry point: builds all simulated institution clients from the real
local datasets and runs the modality-aware federated simulation end-to-end.

Usage:
    python -m src.train_federated
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.data.clinical_dryad import FEATURE_COLUMNS, load_clinical_clients, train_val_test_split
from src.data.ct_dataset import CTDataset, list_ct_files
from src.data.cxr_dataset import CXRDataset, list_cxr_files
from src.federated.client import FederatedClient, compute_class_weights
from src.federated.server import FederatedServer

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cxr_client(name: str, root: str, source: str, image_size: int, batch_size: int, seed: int) -> FederatedClient:
    samples = list_cxr_files(root, source)
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = max(1, int(0.2 * len(samples)))
    val_samples, train_samples = samples[:n_val], samples[n_val:]

    train_ds = CXRDataset(train_samples, image_size=image_size, train=True)
    val_ds = CXRDataset(val_samples, image_size=image_size, train=False)
    weights = compute_class_weights([label for _, label in train_samples])
    return FederatedClient(
        name=name,
        modality="cxr",
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        n_train=len(train_ds),
        class_weights=weights,
    )


def build_clinical_clients(xlsx_path: str, n_clients: int, batch_size: int, seed: int) -> list[FederatedClient]:
    splits = load_clinical_clients(xlsx_path, n_clients=n_clients, seed=seed)
    clients = []
    for i, split in enumerate(splits):
        parts = train_val_test_split(split, seed=seed)
        train_ds = TensorDataset(torch.from_numpy(parts["train"].X), torch.from_numpy(parts["train"].y))
        val_ds = TensorDataset(torch.from_numpy(parts["val"].X), torch.from_numpy(parts["val"].y))
        weights = compute_class_weights(parts["train"].y)
        clients.append(
            FederatedClient(
                name=f"hospital_clinical_{chr(97 + i)}",
                modality="clinical",
                train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
                n_train=len(train_ds),
                class_weights=weights,
            )
        )
    return clients


def build_ct_client(name: str, root: str, size: int, batch_size: int, seed: int) -> FederatedClient:
    samples = list_ct_files(root)
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = max(1, int(0.2 * len(samples)))
    val_samples, train_samples = samples[:n_val], samples[n_val:]

    train_ds = CTDataset(train_samples, size=size)
    val_ds = CTDataset(val_samples, size=size)
    weights = compute_class_weights([label for _, label in train_samples])
    return FederatedClient(
        name=name,
        modality="ct",
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        n_train=len(train_ds),
        class_weights=weights,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = cfg["federated"]["batch_size"]
    embedding_dim = cfg["embedding_dim"]

    clients: list[FederatedClient] = []
    clients.append(
        build_cxr_client(
            "hospital_cxr_montgomery", cfg["paths"]["montgomery_dir"], "montgomery",
            cfg["image"]["cxr_size"], batch_size, cfg["seed"],
        )
    )
    clients.append(
        build_cxr_client(
            "hospital_cxr_shenzhen", cfg["paths"]["shenzhen_dir"], "shenzhen",
            cfg["image"]["cxr_size"], batch_size, cfg["seed"],
        )
    )
    clients.extend(
        build_clinical_clients(cfg["paths"]["dryad_xlsx"], n_clients=2, batch_size=batch_size, seed=cfg["seed"])
    )
    clients.append(
        build_ct_client(
            "hospital_ct", cfg["paths"]["ct_subset_dir"], cfg["image"]["ct_size"], batch_size=4, seed=cfg["seed"],
        )
    )

    clinical_input_dim = len(FEATURE_COLUMNS)

    for c in clients:
        print(f"client={c.name:28s} modality={c.modality:9s} n_train={c.n_train}")

    server = FederatedServer(
        clients=clients,
        embedding_dim=embedding_dim,
        clinical_input_dim=clinical_input_dim,
        device=device,
        local_epochs=cfg["federated"]["local_epochs"],
        lr=cfg["federated"]["lr"],
    )

    num_rounds = args.rounds or cfg["federated"]["num_rounds"]
    history = []
    for r in range(1, num_rounds + 1):
        metrics = server.run_round()
        per_client = {
            k: {"acc": round(v["accuracy"], 4), "f1": round(v["f1"], 4)}
            for k, v in metrics["per_client"].items()
        }
        print(f"[round {r:02d}] macro_acc={metrics['macro_accuracy']:.4f}  {per_client}")
        history.append({"round": r, "macro_accuracy": metrics["macro_accuracy"], "per_client": per_client})

    out_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "federated_history.json", "w") as f:
        json.dump(history, f, indent=2)

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"encoders": server.global_encoders, "fusion": server.global_fusion},
        ckpt_dir / "federated_global.pt",
    )
    print(f"Saved history to {out_dir / 'federated_history.json'} and checkpoint to {ckpt_dir / 'federated_global.pt'}")


if __name__ == "__main__":
    main()
