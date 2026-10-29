"""Orchestrates federated rounds: maintains one global encoder per modality plus
one shared global fusion head, and runs modality-aware FedAvg aggregation.
"""
from __future__ import annotations

import torch

from src.federated.aggregate import weighted_average_state_dicts
from src.federated.client import FederatedClient, evaluate_client, local_train
from src.models.fusion import MODALITIES
from src.models.model_factory import build_encoder, build_fusion


class FederatedServer:
    def __init__(
        self,
        clients: list[FederatedClient],
        embedding_dim: int,
        clinical_input_dim: int,
        device: torch.device,
        local_epochs: int = 1,
        lr: float = 3e-4,
    ):
        self.clients = clients
        self.embedding_dim = embedding_dim
        self.clinical_input_dim = clinical_input_dim
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr

        self.global_encoders = {
            m: build_encoder(m, embedding_dim, clinical_input_dim).state_dict() for m in MODALITIES
        }
        self.global_fusion = build_fusion(embedding_dim).state_dict()

    def run_round(self) -> dict:
        encoder_updates = {m: [] for m in MODALITIES}
        fusion_updates = []

        for client in self.clients:
            enc_state, fus_state = local_train(
                client,
                self.global_encoders[client.modality],
                self.global_fusion,
                self.embedding_dim,
                self.clinical_input_dim,
                self.device,
                epochs=self.local_epochs,
                lr=self.lr,
            )
            encoder_updates[client.modality].append((enc_state, client.n_train))
            fusion_updates.append((fus_state, client.n_train))

        for m in MODALITIES:
            if encoder_updates[m]:
                self.global_encoders[m] = weighted_average_state_dicts(encoder_updates[m])
        self.global_fusion = weighted_average_state_dicts(fusion_updates)

        return self.evaluate()

    @torch.no_grad()
    def evaluate(self) -> dict:
        per_client = {}
        for client in self.clients:
            per_client[client.name] = evaluate_client(
                client,
                self.global_encoders[client.modality],
                self.global_fusion,
                self.embedding_dim,
                self.clinical_input_dim,
                self.device,
            )
        macro_accuracy = sum(r["accuracy"] for r in per_client.values()) / max(len(per_client), 1)
        return {"per_client": per_client, "macro_accuracy": macro_accuracy}
