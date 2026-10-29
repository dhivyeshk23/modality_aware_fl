"""A FederatedClient simulates one institution: it only ever has ONE modality of
data, and only ever trains/evaluates with that single modality's encoder + the
shared fusion head (other modalities are presented to the fusion head as
"absent").
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from src.models.fusion import MODALITIES
from src.models.model_factory import build_encoder, build_fusion


@dataclass
class FederatedClient:
    name: str
    modality: str
    train_loader: DataLoader
    val_loader: DataLoader
    n_train: int
    test_loader: DataLoader | None = None
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
    loader: DataLoader | None = None,
    threshold: float = 0.5,
) -> dict:
    encoder = build_encoder(client.modality, embedding_dim, clinical_input_dim).to(device)
    fusion = build_fusion(embedding_dim).to(device)
    encoder.load_state_dict(encoder_state)
    fusion.load_state_dict(fusion_state)
    encoder.eval()
    fusion.eval()

    evaluation_loader = loader or client.val_loader
    total, loss_sum = 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_probs, all_labels = [], []
    for x, y in evaluation_loader:
        y = y.to(device)
        logits = _forward_batch(encoder, fusion, client.modality, x, device)
        loss_sum += criterion(logits, y).item()
        total += y.size(0)
        all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    result = binary_metrics(all_labels, all_probs, threshold)
    result.update({"loss": loss_sum / max(total, 1), "probs": all_probs, "labels": all_labels})
    return result


def binary_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict:
    preds = [p >= threshold for p in probabilities]
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and not p)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and not p)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    has_both_classes = len(set(labels)) == 2
    ece = 0.0
    if labels:
        for bin_index in range(10):
            lower, upper = bin_index / 10, (bin_index + 1) / 10
            indices = [
                i for i, p in enumerate(probabilities)
                if lower <= p < upper or (bin_index == 9 and p == 1.0)
            ]
            if indices:
                confidence = sum(probabilities[i] for i in indices) / len(indices)
                observed = sum(labels[i] for i in indices) / len(indices)
                ece += len(indices) / len(labels) * abs(confidence - observed)
    return {
        "n": len(labels),
        "positive_count": sum(labels),
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(labels), 1),
        "balanced_accuracy": balanced_accuracy_score(labels, preds) if labels else 0.0,
        "precision": precision,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": roc_auc_score(labels, probabilities) if has_both_classes else None,
        "auprc": average_precision_score(labels, probabilities) if has_both_classes else None,
        "brier": brier_score_loss(labels, probabilities) if labels else None,
        "ece_10bin": ece if labels else None,
    }


def select_validation_threshold(labels: list[int], probabilities: list[float]) -> float:
    """Select a threshold by validation balanced accuracy; never inspect test labels."""
    if len(set(labels)) < 2:
        return 0.5
    candidates = sorted({0.05, 0.5, 0.95, *probabilities})
    scores = [balanced_accuracy_score(labels, [p >= t for p in probabilities]) for t in candidates]
    best = max(scores)
    return min((t for t, score in zip(candidates, scores) if score == best), key=lambda t: abs(t - 0.5))


@torch.no_grad()
def evaluate_heldout_client(
    client: FederatedClient,
    encoder_state: dict,
    fusion_state: dict,
    embedding_dim: int,
    clinical_input_dim: int,
    device: torch.device,
) -> dict:
    if client.test_loader is None:
        raise ValueError(f"client {client.name} has no held-out test loader")
    validation = evaluate_client(
        client, encoder_state, fusion_state, embedding_dim, clinical_input_dim, device
    )
    threshold = select_validation_threshold(validation["labels"], validation["probs"])
    test = evaluate_client(
        client, encoder_state, fusion_state, embedding_dim, clinical_input_dim, device,
        loader=client.test_loader, threshold=threshold,
    )
    metric_names = (
        "n", "positive_count", "threshold", "accuracy", "balanced_accuracy",
        "precision", "sensitivity", "specificity", "f1", "auroc", "auprc", "brier",
        "ece_10bin",
    )
    result = {key: test[key] for key in metric_names}
    result["confidence_intervals_95"] = bootstrap_confidence_intervals(
        test["labels"], test["probs"], threshold, seed=sum(map(ord, client.name))
    )
    return result


def bootstrap_confidence_intervals(
    labels: list[int], probabilities: list[float], threshold: float,
    seed: int = 42, iterations: int = 500,
) -> dict:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    negative = np.flatnonzero(labels_array == 0)
    positive = np.flatnonzero(labels_array == 1)
    if not len(negative) or not len(positive):
        return {}

    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("balanced_accuracy", "f1", "auroc", "auprc")}
    for _ in range(iterations):
        indices = np.concatenate([
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ])
        metrics = binary_metrics(
            labels_array[indices].tolist(), probabilities_array[indices].tolist(), threshold
        )
        for key in samples:
            value = metrics[key]
            if value is not None:
                samples[key].append(value)
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in samples.items() if values
    }


def compute_class_weights(labels, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights, computed locally from one client's own
    training labels only (never shared raw data across institutions)."""
    counts = torch.bincount(torch.as_tensor(labels), minlength=num_classes).float()
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights
