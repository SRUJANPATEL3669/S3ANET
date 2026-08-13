"""Loader for S3ANet (Xu, Xu, Jiao, Gao & Zhang, IEEE TGRS 2024) — a
spatial-spectral self-attention defense network, explicitly designed to
defend against adversarial attacks in HSI classification (successor to
SACNet, same authors' lineage, acknowledges SACNet/FullyContNet/CCNet).

Same whole-scene FCN input/output contract as SACNet: (1, n_bands, H, W) in,
(1, n_classes, H, W) per-pixel logits out. Architecturally heavier — a
pyramid spatial-attention head (PPM_Spa, criss-cross attention per grid bin)
plus a global spectral transformer block (GST, channel-attention over the
whole spectrum) — but nothing in it breaks autograd, so it's attackable the
same way as SACNet.

Model code not vendored — lives in a clone of the upstream repo at
S3ANET_REPO_DIR (github.com/YichuXu/S3ANet). No pretrained checkpoints ship
with that repo, so this project trains its own (see
scripts/train_s3anet_paviau.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

S3ANET_REPO_DIR = "/home/jayant/projects/S3ANet"


def load_s3anet(
    n_bands: int = 103,
    n_classes: int = 9,
    bins: tuple[int, ...] = (1, 2, 3, 6),
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Returns an untrained S3ANet with the given input/output dims.
    Input format: (1, n_bands, H, W) whole-scene, per-band min-max normalized
    to [0, 1]. Output: (1, n_classes, H, W) per-pixel logits."""
    if not Path(S3ANET_REPO_DIR).is_dir():
        raise FileNotFoundError(
            f"S3ANet repo not found at {S3ANET_REPO_DIR} — clone github.com/YichuXu/S3ANet there first."
        )
    if S3ANET_REPO_DIR not in sys.path:
        sys.path.insert(0, S3ANET_REPO_DIR)
    from Model_S3ANet import S3ANet

    return S3ANet(num_features=n_bands, num_classes=n_classes, bins=list(bins)).to(device)
