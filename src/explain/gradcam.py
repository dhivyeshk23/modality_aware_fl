"""Grad-CAM explainability for the CXR encoder - lets us point at which lung
regions drove the model's TB prediction (feeds the clinician-facing view)."""
from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.encoders import CXREncoder
from src.models.fusion import MissingModalityFusion, MODALITIES


class CXRExplainer:
    """Wraps a trained CXR encoder + fusion head so Grad-CAM sees a single
    end-to-end classifier (required since pytorch-grad-cam expects one model)."""

    class _CombinedModel(torch.nn.Module):
        def __init__(self, encoder: CXREncoder, fusion: MissingModalityFusion, device):
            super().__init__()
            self.encoder = encoder
            self.fusion = fusion
            self.device = device

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            emb = self.encoder(x)
            embeddings = {m: (emb if m == "cxr" else None) for m in MODALITIES}
            return self.fusion(embeddings, batch_size=x.size(0), device=self.device)

    def __init__(self, encoder: CXREncoder, fusion: MissingModalityFusion, device: torch.device):
        self.model = self._CombinedModel(encoder, fusion, device).to(device).eval()
        target_layer = encoder.backbone.layer4[-1]
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def explain(self, image_tensor: torch.Tensor, rgb_image: np.ndarray) -> np.ndarray:
        """image_tensor: (1,3,H,W) normalized input. rgb_image: (H,W,3) float [0,1]
        for overlay. Returns the CAM-overlaid RGB image."""
        grayscale_cam = self.cam(
            input_tensor=image_tensor, targets=[ClassifierOutputTarget(1)]
        )[0]
        return show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
