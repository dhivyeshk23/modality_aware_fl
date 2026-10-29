"""Heuristic CXR quality indicators; these are not a trained quality model."""
from __future__ import annotations

import numpy as np
from PIL import Image


def assess_cxr_quality(image: Image.Image, size: int = 224) -> dict:
    gray = np.asarray(
        image.convert("L").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness_proxy = float(laplacian.var())
    clipped_fraction = float(((gray <= 0.01) | (gray >= 0.99)).mean())

    warnings = []
    if brightness < 0.08 or brightness > 0.92:
        warnings.append("extreme image brightness")
    if contrast < 0.08:
        warnings.append("low image contrast")
    if sharpness_proxy < 0.0001:
        warnings.append("possible blur or low detail")
    if clipped_fraction > 0.2:
        warnings.append("substantial clipped dark/bright area")

    return {
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "sharpness_proxy": sharpness_proxy,
        "clipped_fraction": clipped_fraction,
        "warnings": warnings,
    }