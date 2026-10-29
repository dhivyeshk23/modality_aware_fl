"""Missing-modality-aware fusion: combines per-modality embeddings into a single
patient representation and classifies TB risk. Any modality can be absent (the
whole point of this project - institutions rarely hold every modality) - a
learned "null" embedding stands in for a missing modality, and a gating network
down-weights it automatically relative to whatever modalities ARE present.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODALITIES = ("cxr", "clinical", "ct")


class MissingModalityFusion(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.null_embeddings = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(embedding_dim)) for m in MODALITIES}
        )
        # Gate scores how much each modality should contribute, conditioned on
        # both its embedding and an explicit presence flag (so missing modalities
        # are something the network is told about, not silently zero-filled).
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim + 1, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embeddings: dict[str, torch.Tensor | None], batch_size: int, device) -> torch.Tensor:
        """embeddings: modality name -> (B, embedding_dim) tensor, or None if that
        modality is absent for this entire batch (institution-level absence)."""
        stacked, presence = [], []
        for m in MODALITIES:
            emb = embeddings.get(m)
            if emb is None:
                null = self.null_embeddings[m].unsqueeze(0).expand(batch_size, -1).to(device)
                stacked.append(null)
                presence.append(torch.zeros(batch_size, 1, device=device))
            else:
                stacked.append(emb)
                presence.append(torch.ones(batch_size, 1, device=device))

        stacked = torch.stack(stacked, dim=1)  # (B, M, D)
        presence = torch.stack(presence, dim=1)  # (B, M, 1)

        gate_logits = self.gate(torch.cat([stacked, presence], dim=-1))  # (B, M, 1)
        weights = torch.softmax(gate_logits, dim=1)  # (B, M, 1)
        fused = (stacked * weights).sum(dim=1)  # (B, D)

        return self.classifier(fused)
