"""CXR modality data source: Montgomery & Shenzhen chest X-ray sets.

Both datasets share the NLM naming convention `<id>_<label>.png` where label is
0 = normal, 1 = TB. Each dataset is treated as one federated client (institution).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _label_from_name(path: Path) -> int:
    stem = path.stem  # e.g. MCUCXR_0104_1
    return int(stem.rsplit("_", 1)[-1])


def list_cxr_files(root: str | Path, source: str) -> list[tuple[Path, int]]:
    root = Path(root)
    if source == "montgomery":
        img_dir = root / "CXR_png"
    elif source == "shenzhen":
        img_dir = root / "images" / "images"
    else:
        raise ValueError(f"unknown CXR source: {source}")
    files = sorted(img_dir.glob("*.png"))
    return [(f, _label_from_name(f)) for f in files]


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Grayscale(num_output_channels=3), transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=7)]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


@dataclass
class CXRSample:
    path: Path
    label: int


class CXRDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], image_size: int = 224, train: bool = True):
        self.samples = samples
        self.transform = build_transform(image_size, train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label
