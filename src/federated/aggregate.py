"""FedAvg-style weight aggregation, modality-aware: encoders are only ever
averaged across clients that hold the matching modality; the shared fusion
head is averaged across every client regardless of modality.
"""
from __future__ import annotations

import copy

import torch


def weighted_average_state_dicts(
    updates: list[tuple[dict, int]]
) -> dict:
    """updates: list of (state_dict, n_samples). Returns the sample-weighted mean,
    preserving each tensor's original dtype (so integer buffers like
    `num_batches_tracked` don't get silently turned into floats)."""
    if not updates:
        raise ValueError("no updates to aggregate")
    total = sum(n for _, n in updates)
    keys = updates[0][0].keys()
    avg = {}
    for key in keys:
        dtype = updates[0][0][key].dtype
        acc = torch.zeros_like(updates[0][0][key], dtype=torch.float32)
        for state_dict, n in updates:
            acc += state_dict[key].to(torch.float32) * (n / total)
        avg[key] = acc.to(dtype)
    return avg


def clone_state_dict(state_dict: dict) -> dict:
    return copy.deepcopy(state_dict)
