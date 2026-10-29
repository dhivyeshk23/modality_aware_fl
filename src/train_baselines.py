"""Phase 4: baselines to compare against the modality-aware federated approach.

- centralized: pools every client's training data into ONE joint training loop
  each "round" (upper-bound reference; not realistic since it violates the
  privacy constraint the whole project exists to avoid, but useful as a ceiling).
- local_only: every institution trains in total isolation, never sharing any
  weights (lower-bound reference - shows why federation helps at all).
- plain_fedavg: same federated loop as Phase 3, but with UNWEIGHTED (equal)
  averaging instead of sample-size-weighted averaging - the textbook FedAvg
  that ignores how unevenly sized/available each institution's data/modality is.

Usage:
    python -m src.train_baselines --rounds 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.federated.aggregate import weighted_average_state_dicts
from src.federated.client import evaluate_client, local_train
from src.federated.server import FederatedServer
from src.models.fusion import MODALITIES
from src.models.model_factory import build_encoder, build_fusion
from src.train_federated import (
    build_clinical_clients,
    build_ct_client,
    build_cxr_client,
    set_seed,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def build_clients(cfg):
    clients = [
        build_cxr_client(
            "hospital_cxr_montgomery", cfg["paths"]["montgomery_dir"], "montgomery",
            cfg["image"]["cxr_size"], cfg["federated"]["batch_size"], cfg["seed"],
        ),
        build_cxr_client(
            "hospital_cxr_shenzhen", cfg["paths"]["shenzhen_dir"], "shenzhen",
            cfg["image"]["cxr_size"], cfg["federated"]["batch_size"], cfg["seed"],
        ),
    ]
    clients.extend(
        build_clinical_clients(cfg["paths"]["dryad_xlsx"], n_clients=2,
                                batch_size=cfg["federated"]["batch_size"], seed=cfg["seed"])
    )
    clients.append(
        build_ct_client("hospital_ct", cfg["paths"]["ct_subset_dir"], cfg["image"]["ct_size"],
                         batch_size=4, seed=cfg["seed"])
    )
    return clients


def unweighted_average_state_dicts(updates: list[tuple[dict, int]]) -> dict:
    """Same as weighted_average_state_dicts but every client counts equally,
    regardless of how much data it actually holds - the naive FedAvg baseline."""
    return weighted_average_state_dicts([(state, 1) for state, _ in updates])


def run_plain_fedavg(clients, cfg, device) -> list[dict]:
    server = FederatedServer(
        clients=clients,
        embedding_dim=cfg["embedding_dim"],
        clinical_input_dim=clients_clinical_dim(),
        device=device,
        local_epochs=cfg["federated"]["local_epochs"],
        lr=cfg["federated"]["lr"],
    )
    # monkey-patch: swap in unweighted aggregation for this baseline only
    history = []
    for r in range(1, cfg["federated"]["num_rounds"] + 1):
        encoder_updates = {m: [] for m in MODALITIES}
        fusion_updates = []
        for client in server.clients:
            enc_state, fus_state = local_train(
                client, server.global_encoders[client.modality], server.global_fusion,
                server.embedding_dim, server.clinical_input_dim, device,
                epochs=server.local_epochs, lr=server.lr,
            )
            encoder_updates[client.modality].append((enc_state, client.n_train))
            fusion_updates.append((fus_state, client.n_train))
        for m in MODALITIES:
            if encoder_updates[m]:
                server.global_encoders[m] = unweighted_average_state_dicts(encoder_updates[m])
        server.global_fusion = unweighted_average_state_dicts(fusion_updates)
        metrics = server.evaluate()
        history.append(summarize(r, metrics))
        print(f"[plain_fedavg][round {r:02d}] {history[-1]}")
    return history


def run_local_only(clients, cfg, device) -> list[dict]:
    embedding_dim = cfg["embedding_dim"]
    clinical_dim = clients_clinical_dim()
    # each client keeps its OWN encoder+fusion state across rounds - no sharing at all
    local_encoders = {c.name: build_encoder(c.modality, embedding_dim, clinical_dim).state_dict() for c in clients}
    local_fusions = {c.name: build_fusion(embedding_dim).state_dict() for c in clients}

    history = []
    for r in range(1, cfg["federated"]["num_rounds"] + 1):
        results = {}
        for client in clients:
            enc_state, fus_state = local_train(
                client, local_encoders[client.name], local_fusions[client.name],
                embedding_dim, clinical_dim, device,
                epochs=cfg["federated"]["local_epochs"], lr=cfg["federated"]["lr"],
            )
            local_encoders[client.name] = enc_state
            local_fusions[client.name] = fus_state
            results[client.name] = evaluate_client(
                client, enc_state, fus_state, embedding_dim, clinical_dim, device
            )
        acc = sum(v["accuracy"] for v in results.values()) / max(len(results), 1)
        entry = summarize(r, {"per_client": results, "macro_accuracy": acc})
        history.append(entry)
        print(f"[local_only][round {r:02d}] {entry}")
    return history


def run_centralized(clients, cfg, device) -> list[dict]:
    """Pools every batch from every client into one shared optimizer step per
    round (same modality-aware architecture, just no federation boundary)."""
    embedding_dim = cfg["embedding_dim"]
    clinical_dim = clients_clinical_dim()
    encoders = {m: build_encoder(m, embedding_dim, clinical_dim) for m in MODALITIES}
    fusion = build_fusion(embedding_dim)
    for enc in encoders.values():
        enc.to(device)
    fusion.to(device)

    params = list(fusion.parameters())
    for enc in encoders.values():
        params += list(enc.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["federated"]["lr"])
    criterion_by_client = {
        c.name: torch.nn.CrossEntropyLoss(
            weight=c.class_weights.to(device) if c.class_weights is not None else None
        )
        for c in clients
    }

    history = []
    for r in range(1, cfg["federated"]["num_rounds"] + 1):
        for enc in encoders.values():
            enc.train()
        fusion.train()
        for client in clients:
            for x, y in client.train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                emb = encoders[client.modality](x)
                embeddings = {m: (emb if m == client.modality else None) for m in MODALITIES}
                logits = fusion(embeddings, batch_size=x.size(0), device=device)
                loss = criterion_by_client[client.name](logits, y)
                loss.backward()
                optimizer.step()

        results = {}
        for enc in encoders.values():
            enc.eval()
        fusion.eval()
        for client in clients:
            results[client.name] = evaluate_client(
                client, encoders[client.modality].state_dict(), fusion.state_dict(),
                embedding_dim, clinical_dim, device,
            )
        acc = sum(v["accuracy"] for v in results.values()) / max(len(results), 1)
        entry = summarize(r, {"per_client": results, "macro_accuracy": acc})
        history.append(entry)
        print(f"[centralized][round {r:02d}] {entry}")
    return history


def summarize(round_idx: int, metrics: dict) -> dict:
    per_client = {
        k: {"acc": round(v["accuracy"], 4), "f1": round(v["f1"], 4)} for k, v in metrics["per_client"].items()
    }
    return {"round": round_idx, "macro_accuracy": metrics["macro_accuracy"], "per_client": per_client}


_CLINICAL_DIM = None


def clients_clinical_dim() -> int:
    global _CLINICAL_DIM
    if _CLINICAL_DIM is None:
        from src.data.clinical_dryad import FEATURE_COLUMNS

        _CLINICAL_DIM = len(FEATURE_COLUMNS)
    return _CLINICAL_DIM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument(
        "--which", choices=["centralized", "local_only", "plain_fedavg", "all"], default="all"
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if args.rounds:
        cfg["federated"]["num_rounds"] = args.rounds

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "baselines_history.json"
    all_history = {}
    if history_path.exists():
        with open(history_path) as f:
            all_history = json.load(f)
    if args.which in ("centralized", "all"):
        all_history["centralized"] = run_centralized(build_clients(cfg), cfg, device)
    if args.which in ("local_only", "all"):
        all_history["local_only"] = run_local_only(build_clients(cfg), cfg, device)
    if args.which in ("plain_fedavg", "all"):
        all_history["plain_fedavg"] = run_plain_fedavg(build_clients(cfg), cfg, device)

    with open(history_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"Saved baseline histories to {history_path}")


if __name__ == "__main__":
    main()
