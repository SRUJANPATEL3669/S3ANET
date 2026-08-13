"""Loader for SACNet (Xu, Du & Zhang, IEEE TIP 2021) — a context-aware
defense baseline named in H_2 §1.C, used here as a Tier 2+ classifier target
for RTAA (a fully-convolutional, whole-scene architecture with a
self-attention context module, quite different from HybridSN/SpectralFormer's
patch-/pixel-wise pipelines).

The model code is not vendored — it lives in a clone of the upstream repo at
SACNET_REPO_DIR (github.com/YonghaoXu/SACNet). No pretrained checkpoints ship
with that repo, so this project trains its own (see
scripts/train_sacnet_paviau.py) rather than loading someone else's weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SACNET_REPO_DIR = "/home/jayant/projects/SACNet"


def load_sacnet(n_bands: int = 103, n_classes: int = 9, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Returns an untrained SACNet with the given input/output dims.
    Input format: (1, n_bands, H, W) whole-scene, per-band min-max normalized
    to [0, 1]. Output: (1, n_classes, H, W) per-pixel logits."""
    if not Path(SACNET_REPO_DIR).is_dir():
        raise FileNotFoundError(
            f"SACNet repo not found at {SACNET_REPO_DIR} — clone github.com/YonghaoXu/SACNet there first."
        )
    if SACNET_REPO_DIR not in sys.path:
        sys.path.insert(0, SACNET_REPO_DIR)
    from Models import SACNet

    return SACNet(num_features=n_bands, num_classes=n_classes).to(device)
