"""Phase 5: uncertainty estimation (MC-Dropout) + two-tier output views.

Clinicians get the full probability + uncertainty + modality-availability
picture; patients get a deliberately bounded, cautious message that always
defers to a professional rather than stating a diagnosis.

Usage:
    python -m src.predict --modality cxr --image path/to/xray.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.data.clinical_dryad import FEATURE_COLUMNS
from src.data.cxr_dataset import build_transform
from src.data.image_quality import assess_cxr_quality
from src.models.fusion import MODALITIES
from src.models.model_factory import build_encoder, build_fusion

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Keep the model in eval() mode (frozen BatchNorm) but re-enable Dropout
    layers so repeated forward passes sample different sub-networks."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(encoder, fusion, modality: str, x: torch.Tensor, device, n_samples: int = 20):
    encoder.eval()
    fusion.eval()
    enable_mc_dropout(fusion)  # only the fusion head's classifier has Dropout

    x = x.to(device)
    probs = []
    for _ in range(n_samples):
        emb = encoder(x)
        embeddings = {m: (emb if m == modality else None) for m in MODALITIES}
        logits = fusion(embeddings, batch_size=x.size(0), device=device)
        probs.append(F.softmax(logits, dim=1)[:, 1])
    probs = torch.stack(probs, dim=0)  # (n_samples, B)
    return probs.mean(dim=0), probs.std(dim=0)


def clinician_view(prob_mean: float, prob_std: float, modality: str, quality: dict | None = None) -> str:
    lines = [
        "[Clinician view]",
        f"  Modality used: {modality}",
        f"  Estimated TB-positive probability: {prob_mean:.3f}",
        f"  Model uncertainty (MC-Dropout std over 20 passes): {prob_std:.3f}",
    ]
    if quality is not None:
        flags = ", ".join(quality["warnings"]) or "no threshold flags"
        lines.extend([
            f"  Heuristic image-quality flags: {flags}",
            f"  Contrast={quality['contrast_std']:.3f}; sharpness proxy={quality['sharpness_proxy']:.6f}",
        ])
    lines.extend([
        "  Note: single-modality input - treat as a screening signal, not a diagnosis.",
        "  Recommend confirmatory testing (e.g. Xpert MTB/RIF) regardless of score.",
    ])
    return "\n".join(lines)


def patient_view(prob_mean: float, prob_std: float) -> str:
    if prob_std > 0.15:
        screening = "This experimental screening result is uncertain."
    elif prob_mean >= 0.5:
        screening = "This experimental screening result suggests further evaluation may be appropriate."
    else:
        screening = "This model did not produce a high screening score, but that cannot rule out TB."
    return (
        "[Patient view]\n"
        f"  {screening}\n"
        "  This research prototype has not been clinically validated and cannot diagnose\n"
        "  or rule out tuberculosis. Please consult a qualified healthcare professional,\n"
        "  especially if you have symptoms or have been exposed to TB."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["cxr", "clinical", "ct"], required=True)
    parser.add_argument("--image", type=str, help="path to a CXR png (if --modality cxr)")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint or Path(cfg["paths"]["checkpoints_dir"]) / "federated_global.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    embedding_dim = cfg["embedding_dim"]
    clinical_dim = len(FEATURE_COLUMNS)
    encoder = build_encoder(args.modality, embedding_dim, clinical_dim).to(device)
    encoder.load_state_dict(ckpt["encoders"][args.modality])
    fusion = build_fusion(embedding_dim).to(device)
    fusion.load_state_dict(ckpt["fusion"])

    if args.modality == "cxr":
        if not args.image:
            raise SystemExit("--image is required for --modality cxr")
        transform = build_transform(cfg["image"]["cxr_size"], train=False)
        img = Image.open(args.image).convert("L")
        quality = assess_cxr_quality(img, cfg["image"]["cxr_size"])
        x = transform(img).unsqueeze(0)
    else:
        raise SystemExit("this CLI currently only demos the cxr modality end-to-end")

    mean, std = mc_dropout_predict(encoder, fusion, args.modality, x, device)
    print(clinician_view(mean.item(), std.item(), args.modality, quality if args.modality == "cxr" else None))
    print()
    print(patient_view(mean.item(), std.item()))


if __name__ == "__main__":
    main()
