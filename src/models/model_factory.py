"""Factory for building fresh encoder/fusion module instances with consistent
hyperparameters - used whenever a client needs a local copy of the global model.
"""
from __future__ import annotations

from src.models.encoders import CTEncoder, ClinicalEncoder, CXREncoder
from src.models.fusion import MissingModalityFusion


def build_encoder(modality: str, embedding_dim: int, clinical_input_dim: int | None = None):
    if modality == "cxr":
        return CXREncoder(embedding_dim=embedding_dim, pretrained=True)
    if modality == "clinical":
        assert clinical_input_dim is not None
        return ClinicalEncoder(input_dim=clinical_input_dim, embedding_dim=embedding_dim)
    if modality == "ct":
        return CTEncoder(embedding_dim=embedding_dim)
    raise ValueError(f"unknown modality: {modality}")


def build_fusion(embedding_dim: int, num_classes: int = 2) -> MissingModalityFusion:
    return MissingModalityFusion(embedding_dim=embedding_dim, num_classes=num_classes)
