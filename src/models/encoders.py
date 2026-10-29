"""Modality-specific encoders. Each encoder maps its raw input to a fixed-size
embedding vector of dimension `embedding_dim`, so the fusion module never needs
to know about modality-specific architectures.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CXREncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.proj(feats)


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CTEncoder(nn.Module):
    """Small 3D CNN - kept lightweight to fit a 6GB GPU alongside the 2D CXR encoder."""

    def __init__(self, embedding_dim: int = 128, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 64 -> 32
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 32 -> 16
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 16 -> 8
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x).flatten(1)
        return self.proj(feats)
