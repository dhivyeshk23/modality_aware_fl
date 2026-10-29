"""CT modality data source: small local subset of the Mycobacterial CT Imaging
Dataset (TBNifTI = TB positive, NTMNiFTi = non-tuberculous mycobacteria / TB
negative look-alike). Volumes are resampled to a fixed small cube to fit a 6GB
GPU.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CLASS_TO_LABEL = {"TB": 1, "NTM": 0}


def list_ct_files(root: str | Path) -> list[tuple[Path, int]]:
    root = Path(root)
    samples = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for f in sorted(class_dir.rglob("*.nii")):
            samples.append((f, label))
    return samples


@lru_cache(maxsize=64)
def _load_volume(path: str, size: int) -> torch.Tensor:
    vol = nib.load(str(path)).get_fdata(dtype=np.float32)
    # Robust windowing: clip to the 1st-99th percentile range, then min-max normalize.
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip(vol, lo, hi)
    denom = (hi - lo) if hi > lo else 1.0
    vol = (vol - lo) / denom
    tensor = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    tensor = F.interpolate(tensor, size=(size, size, size), mode="trilinear", align_corners=False)
    return tensor.squeeze(0)  # (1,size,size,size)


class CTDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], size: int = 64):
        self.samples = samples
        self.size = size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        vol = _load_volume(str(path), self.size).clone()
        return vol, label
