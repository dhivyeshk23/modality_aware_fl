"""Evaluate CXR robustness to simple image-quality degradations on the held-out split.

These transformations are stress tests, not a clinical image-quality benchmark.
Run after `src.train_federated` has produced the global checkpoint.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from src.data.clinical_dryad import FEATURE_COLUMNS
from src.federated.client import evaluate_heldout_client
from src.models.model_factory import build_encoder, build_fusion
from src.train_federated import build_cxr_client

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


class PerturbedCXR(Dataset):
    def __init__(self, base: Dataset, mode: str, level: float):
        self.base = base
        self.mode = mode
        self.level = level

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        if self.mode == "gaussian_blur":
            image = TF.gaussian_blur(image, kernel_size=15, sigma=self.level)
        elif self.mode == "gaussian_noise":
            generator = torch.Generator().manual_seed(index)
            image = image + torch.randn(image.shape, generator=generator) * self.level
        return image, label


CONDITIONS = (
    ("clean", "clean", 0.0),
    ("blur_1", "gaussian_blur", 1.0),
    ("blur_2", "gaussian_blur", 2.0),
    ("blur_4", "gaussian_blur", 4.0),
    ("noise_0.05", "gaussian_noise", 0.05),
    ("noise_0.15", "gaussian_noise", 0.15),
    ("noise_0.30", "gaussian_noise", 0.30),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/study_seed42.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    embedding_dim = cfg["embedding_dim"]
    clinical_dim = len(FEATURE_COLUMNS)

    sources = (
        ("hospital_cxr_montgomery", cfg["paths"]["montgomery_dir"], "montgomery"),
        ("hospital_cxr_shenzhen", cfg["paths"]["shenzhen_dir"], "shenzhen"),
    )
    results = {}
    for name, root, source in sources:
        client = build_cxr_client(
            name, root, source, cfg["image"]["cxr_size"],
            cfg["federated"]["batch_size"], args.seed,
        )
        encoder = checkpoint["encoders"]["cxr"]
        fusion = checkpoint["fusion"]
        results[name] = {}
        for label, mode, level in CONDITIONS:
            test_loader = client.test_loader
            if mode != "clean":
                dataset = PerturbedCXR(test_loader.dataset, mode, level)
                test_loader = DataLoader(
                    dataset, batch_size=cfg["federated"]["batch_size"], shuffle=False
                )
            stressed_client = replace(client, test_loader=test_loader)
            results[name][label] = evaluate_heldout_client(
                stressed_client, encoder, fusion, embedding_dim, clinical_dim, device
            )

    output_path = Path(cfg["paths"]["outputs_dir"]) / "cxr_quality_stress.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    for name, modes in results.items():
        print(name)
        for mode, metrics in modes.items():
            print(
                f"  {mode:14s} balanced_acc={metrics['balanced_accuracy']:.3f} "
                f"f1={metrics['f1']:.3f} auroc={metrics['auroc']:.3f}"
            )
    print(f"Saved stress results to {output_path}")


if __name__ == "__main__":
    main()
