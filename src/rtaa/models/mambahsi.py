"""Loader for MambaHSI (Li, Luo, Zhang, Wang & Du, IEEE TGRS 2024) — the
first image-level HSI classifier built on a State Space Model (Mamba), and
the "SSMADNet/Mamba" family target named in the RTAA project roadmap.

Whole-scene input like SACNet/S3ANet, but the model does NOT internally
upsample back to full resolution — it downsamples 3x (avgpool stride 2 after
each Mamba block) and returns logits at H/8 x W/8. Callers must
`F.interpolate` back to full size before computing loss/predictions (see
`rtaa/attacks/mambahsi_attack.py`), matching the upstream repo's own
`utils.Loss.head_loss` / eval code.

Model code not vendored — lives in a clone of the upstream repo at
MAMBAHSI_REPO_DIR (github.com/li-yapeng/MambaHSI). Requires the real
`mamba-ssm` package with compiled CUDA selective-scan kernels (CPU inference
is not supported by that package); see scripts/train_mambahsi_paviau.py for
the install notes. No pretrained checkpoints ship with the repo — its own
train_MambaHSI.py trains from scratch too (10-seed loop by default; RTAA
trains a single run instead, see that script).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

MAMBAHSI_REPO_DIR = "/home/jayant/projects/MambaHSI"


class UpsampledMambaHSI(nn.Module):
    """Wraps MambaHSI to upsample its H/8 x W/8 logits back to input
    resolution, so it presents the same (1, n_classes, H, W) I/O contract as
    SACNet/S3ANet — lets `SACNetRTAAAttack` attack it unchanged rather than
    needing a third near-duplicate whole-scene attack module."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        return nn.functional.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=True)


def load_mambahsi(
    n_bands: int = 103,
    n_classes: int = 9,
    hidden_dim: int = 128,
    device: str | torch.device = "cuda:0",
) -> torch.nn.Module:
    """Returns an untrained MambaHSI. Input: (1, n_bands, H, W), per-band
    2nd-98th-percentile-stretched to [0, 1] (see `HSICommonUtils.ImageStretching`
    upstream — replicate that exact normalization for consistency with any
    checkpoint trained via this project's scripts). Output: (1, n_classes,
    H/8, W/8) — needs upsampling to (H, W) before use."""
    if not Path(MAMBAHSI_REPO_DIR).is_dir():
        raise FileNotFoundError(
            f"MambaHSI repo not found at {MAMBAHSI_REPO_DIR} — clone github.com/li-yapeng/MambaHSI there first."
        )
    if MAMBAHSI_REPO_DIR not in sys.path:
        sys.path.insert(0, MAMBAHSI_REPO_DIR)
    from model.MambaHSI import MambaHSI

    return MambaHSI(in_channels=n_bands, num_classes=n_classes, hidden_dim=hidden_dim).to(device)
