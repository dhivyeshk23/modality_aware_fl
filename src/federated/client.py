"""A FederatedClient simulates one institution: it only ever has ONE modality of
data, and only ever trains/evaluates with that single modality's encoder + the
shared fusion head (other modalities are presented to the fusion head as
"absent").
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.fusion import MODALITIES
from src.models.model_factory import build_encoder, build_fusion


@dataclass
class FederatedClient:
    name: str
    modality: str
    train_loader: DataLoader
    val_loader: DataLoader
    n_train: int
    class_weights: torch.Tensor | None = None  # inverse-frequency weights, computed locally


def _forward_batch(encoder, fusion, modality, x, device):
    x = x.to(device)
    emb = encoder(x)
    embeddings = {m: (emb if m == modality else None) for m in MODALITIES}
    logits = fusion(embeddings, batch_size=x.size(0), device=device)
    return logits


def local_train(
    client: FederatedClient,
    global_encoder_state: dict,
    global_fusion_state: dict,
    embedding_dim: int,
    clinical_input_dim: int,
    device: torch.device,
    epochs: int = 1,
    lr: float = 3e-4,
) -> tuple[dict, dict]:
    encoder = build_encoder(client.modality, embedding_dim, clinical_input_dim).to(device)
    fusion = build_fusion(embedding_dim).to(device)
    encoder.load_state_dict(global_encoder_state)
    fusion.load_state_dict(global_fusion_state)

    if client.modality == "cxr":
        for name, parameter in encoder.backbone.named_parameters():
            parameter.requires_grad = name.startswith("layer4")

    encoder.train()
    fusion.train()
    optimizer = torch.optim.Adam(
        [p for p in encoder.parameters() if p.requires_grad] + list(fusion.parameters()), lr=lr
    )
    weight = client.class_weights.to(device) if client.class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)

    for _ in range(epochs):
        for x, y in client.train_loader:
            y = y.to(device)
            optimizer.zero_grad()
            logits = _forward_batch(encoder, fusion, client.modality, x, device)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    return encoder.state_dict(), fusion.state_dict()


@torch.no_grad()
def evaluate_client(
    client: FederatedClient,
    encoder_state: dict,
    fusion_state: dict,
    embedding_dim: int,
    clinical_input_dim: int,
    device: torch.device,
) -> dict:
    encoder = build_encoder(client.modality, embedding_dim, clinical_input_dim).to(device)
    fusion = build_fusion(embedding_dim).to(device)
    encoder.load_state_dict(encoder_state)
    fusion.load_state_dict(fusion_state)
    encoder.eval()
    fusion.eval()

    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_probs, all_labels = [], []
    for x, y in client.val_loader:
        y = y.to(device)
        logits = _forward_batch(encoder, fusion, client.modality, x, device)
        loss_sum += criterion(logits, y).item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    f1 = _binary_f1(all_labels, [p >= 0.5 for p in all_probs])
    return {
        "n": total,
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "f1": f1,
        "probs": all_probs,
        "labels": all_labels,
    }


def _binary_f1(labels: list[int], preds: list[bool]) -> float:
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and not p)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def compute_class_weights(labels, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights, computed locally from one client's own
    training labels only (never shared raw data across institutions)."""
    counts = torch.bincount(torch.as_tensor(labels), minlength=num_classes).float()
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights
