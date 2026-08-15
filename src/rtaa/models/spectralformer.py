"""Loader for the official SpectralFormer ViT (Hong et al., TGRS 2022), used
as a Tier 2 (transformer) classifier target for RTAA, per the project's
two-tier validation strategy (Tier 1 = HybridSN, already attacked
successfully; Tier 2 = ViT/Mamba).

The model code and pretrained checkpoints are not vendored in this repo —
they live in the cloned upstream repo at SPECTRALFORMER_REPO_DIR (GPLv3
licensed, github.com/danfenghong/IEEE_TGRS_SpectralFormer). This module only
imports and configures it; no model code is duplicated here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

SPECTRALFORMER_REPO_DIR = os.environ.get("SPECTRALFORMER_REPO_DIR", "/content/SpectralFormer")

# Hyperparameters and checkpoint mapping straight from the repo's README
# ("How to use it?" section) — must match exactly for state_dict to load.
# "pixelwise": patches=1, band_patches=1, mode='ViT' — plain ViT, full spectrum,
#   no spatial patch, no group-wise spectral embedding.
# "patchwise": patches=7, band_patches=3 (7 for Pavia), mode='CAF' — the actual
#   SpectralFormer architecture (group-wise spectral embedding + cross-layer
#   adaptive fusion).
_VARIANT_CONFIGS: dict[str, dict] = {
    "pixelwise": {"image_size": 1, "near_band": 1, "dim": 64, "depth": 5, "heads": 4,
                  "mlp_dim": 8, "dropout": 0.1, "emb_dropout": 0.1, "mode": "ViT"},
    "patchwise": {"image_size": 7, "near_band": 3, "dim": 64, "depth": 5, "heads": 4,
                  "mlp_dim": 8, "dropout": 0.1, "emb_dropout": 0.1, "mode": "CAF"},
}
_CHECKPOINTS: dict[str, dict[str, tuple]] = {
    "pixelwise": {
        "Indian": ("ViT_indian.pt", 200, 16),
        "Pavia": ("ViT_pavia.pt", 103, 9),
        "Houston": ("ViT_houston.pt", 144, 15),  # this repo's own data, not the SAFER mirror
    },
    "patchwise": {
        "Indian": ("SpectralFormer_patch_indian.pt", 200, 16),
        "Pavia": ("SpectralFormer_patch_pavia.pt", 103, 9),
        "Houston": ("SpectralFormer_patch_houston.pt", 144, 15),
    },
}
# Pavia's patch-wise checkpoint used band_patches=7, not the 3 used everywhere else.
_PAVIA_PATCHWISE_NEAR_BAND = 7


def load_spectralformer_vit(
    dataset: str, variant: str = "pixelwise", device: str | torch.device = "cpu"
) -> torch.nn.Module:
    """Loads a pretrained SpectralFormer checkpoint for `dataset` in
    {"Indian", "Pavia", "Houston"}, `variant` in {"pixelwise", "patchwise"}.

    pixelwise input format: (B, n_bands, 1), per-band min-max normalized to [0, 1].
    patchwise input format: (B, n_bands, patch*patch*band_patch) — build with
    `gain_neighborhood_band` below from a (B, 7, 7, n_bands) spatial patch.
    """
    if not Path(SPECTRALFORMER_REPO_DIR).is_dir():
        raise FileNotFoundError(
            f"SpectralFormer repo not found at {SPECTRALFORMER_REPO_DIR} — clone "
            "github.com/danfenghong/IEEE_TGRS_SpectralFormer there first."
        )
    if SPECTRALFORMER_REPO_DIR not in sys.path:
        sys.path.insert(0, SPECTRALFORMER_REPO_DIR)
    from vit_pytorch import ViT  # type: ignore

    if variant not in _VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant {variant!r}, expected one of {list(_VARIANT_CONFIGS)}")
    if dataset not in _CHECKPOINTS[variant]:
        raise ValueError(f"Unknown dataset {dataset!r}, expected one of {list(_CHECKPOINTS[variant])}")
    ckpt_name, n_bands, n_classes = _CHECKPOINTS[variant][dataset]

    config = dict(_VARIANT_CONFIGS[variant])
    if variant == "patchwise" and dataset == "Pavia":
        config["near_band"] = _PAVIA_PATCHWISE_NEAR_BAND

    model = ViT(num_patches=n_bands, num_classes=n_classes, **config).to(device)
    state_dict = torch.load(Path(SPECTRALFORMER_REPO_DIR) / "log" / ckpt_name, map_location=device)
    model.load_state_dict(state_dict)
    return model


def gain_neighborhood_band(x: torch.Tensor, band_patch: int) -> torch.Tensor:
    """Differentiable port of the repo's `gain_neighborhood_band` (demo.py).
    Builds SpectralFormer's group-wise spectral embedding input.

    x: (B, patch, patch, n_bands) spatial patch, e.g. from a 7x7 neighborhood.
    Returns: (B, n_bands, patch*patch*band_patch), the CAF/patchwise input format.

    Verified equivalent to the original numpy implementation (which builds
    each spectral "neighbor group" via explicit index copies) by noting each
    group is exactly a circular shift of the band axis: group i (i=1..nn) is
    `roll(+i)`, the center group is unshifted, and group i on the other side
    is `roll(-i)`, where nn = band_patch // 2. `torch.roll` is a permutation,
    so this is fully differentiable — this is what makes RTAA attackable
    against the real (not just plain-ViT) SpectralFormer architecture.
    """
    b, p1, p2, n_bands = x.shape
    x_flat = x.reshape(b, p1 * p2, n_bands)
    nn = band_patch // 2

    blocks = [torch.roll(x_flat, shifts=i, dims=-1) for i in range(1, nn + 1)]
    blocks.append(x_flat)
    blocks += [torch.roll(x_flat, shifts=-i, dims=-1) for i in range(1, nn + 1)]

    x_band = torch.cat(blocks, dim=1)  # (B, band_patch*patch*patch, n_bands)
    return x_band.transpose(1, 2)  # (B, n_bands, band_patch*patch*patch)
