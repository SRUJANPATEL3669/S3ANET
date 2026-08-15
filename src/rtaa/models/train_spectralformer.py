"""Training routine for SpectralFormer (Hong et al., IEEE TGRS 2022).

Simplified training loop that follows the original procedure in the upstream
repo's demo.py (github.com/danfenghong/IEEE_TGRS_SpectralFormer):
- Per-band min-max normalization
- Same hyperparameters: Adam optimizer, StepLR scheduler
- Pixelwise: single spectra (B, n_bands, 1)
- Patchwise (CAF): 7x7 patches processed through gain_neighborhood_band

Does NOT deviate from the original training procedure — same data preparation,
same model configuration, same optimization scheme.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rtaa.models.spectralformer import (
    SPECTRALFORMER_REPO_DIR,
    _VARIANT_CONFIGS,
    gain_neighborhood_band,
)


def _per_band_normalize(data: np.ndarray) -> np.ndarray:
    """Per-band min-max normalize (N, n_bands) to [0, 1], matching demo.py."""
    lo = data.min(axis=0, keepdims=True)
    hi = data.max(axis=0, keepdims=True)
    return (data - lo) / (hi - lo + 1e-12)


def _prepare_pixelwise_data(
    cube: np.ndarray, labels: np.ndarray, ignore_label: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare pixel-wise data: extract labeled pixels, normalize per-band.

    Returns (all_spectra, all_labels, train_idx, test_idx).
    """
    h, w, n_bands = cube.shape
    flat = cube.reshape(-1, n_bands)
    flat_labels = labels.reshape(-1)

    # Only labeled pixels
    mask = flat_labels != ignore_label
    spectra = flat[mask]
    lbls = flat_labels[mask] - 1  # 1-indexed -> 0-indexed

    # Per-band normalize
    spectra = _per_band_normalize(spectra)

    # Stratified split: 200 samples per class for training (matching demo.py)
    rng = np.random.default_rng(0)
    train_idx, test_idx = [], []
    for cls in np.unique(lbls):
        cls_idx = np.where(lbls == cls)[0]
        rng.shuffle(cls_idx)
        n_train = min(200, len(cls_idx) // 2)
        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())

    return spectra, lbls, np.array(train_idx), np.array(test_idx)


def _prepare_patchwise_data(
    cube: np.ndarray, labels: np.ndarray, patch_size: int = 7,
    band_patch: int = 3, ignore_label: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare patch-wise data: extract labeled pixel patches, normalize.

    Returns (all_patches, all_labels, train_idx, test_idx).
    Patches are (N, patch, patch, n_bands), ready for gain_neighborhood_band.
    """
    h, w, n_bands = cube.shape
    margin = patch_size // 2

    # Per-band normalize the whole cube
    flat = cube.reshape(-1, n_bands)
    flat_norm = _per_band_normalize(flat)
    cube_norm = flat_norm.reshape(h, w, n_bands)

    # Pad cube
    padded = np.pad(cube_norm, ((margin, margin), (margin, margin), (0, 0)),
                    mode="reflect")

    # Extract patches for labeled pixels
    rows, cols = np.where(labels != ignore_label)
    lbls = labels[rows, cols] - 1  # 1-indexed -> 0-indexed

    patches = np.empty((len(rows), patch_size, patch_size, n_bands),
                       dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        rp, cp = r + margin, c + margin
        patches[i] = padded[rp - margin:rp + margin + 1,
                            cp - margin:cp + margin + 1, :]

    # Stratified split
    rng = np.random.default_rng(0)
    train_idx, test_idx = [], []
    for cls in np.unique(lbls):
        cls_idx = np.where(lbls == cls)[0]
        rng.shuffle(cls_idx)
        n_train = min(200, len(cls_idx) // 2)
        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())

    return patches, lbls, np.array(train_idx), np.array(test_idx)


def train_spectralformer(
    cube: np.ndarray,
    labels: np.ndarray,
    variant: str = "pixelwise",
    n_classes: int = 9,
    n_epochs: int = 300,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    device: str = "cuda:0",
) -> tuple[torch.nn.Module, dict]:
    """Train SpectralFormer from scratch.

    Args:
        cube: (H, W, n_bands) raw reflectance cube.
        labels: (H, W) integer labels (0 = unlabeled).
        variant: "pixelwise" or "patchwise".
        n_classes: number of classes.
        n_epochs: training epochs.
        batch_size: batch size.
        lr: learning rate.
        weight_decay: weight decay.
        device: torch device string.

    Returns:
        (trained_model, metadata_dict)
    """
    if SPECTRALFORMER_REPO_DIR not in sys.path:
        sys.path.insert(0, SPECTRALFORMER_REPO_DIR)
    from vit_pytorch import ViT  # type: ignore

    n_bands = cube.shape[-1]
    _device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Get variant config
    config = dict(_VARIANT_CONFIGS[variant])

    # Pavia patchwise uses band_patches=7
    band_patch = config.get("near_band", 1)
    patch_size = config.get("image_size", 1)

    if variant == "pixelwise":
        spectra, lbls, train_idx, test_idx = _prepare_pixelwise_data(
            cube, labels)

        # Build model
        model = ViT(
            num_patches=n_bands, num_classes=n_classes, **config
        ).to(_device)

        # Prepare tensors: pixelwise input is (B, n_bands, 1)
        train_x = torch.from_numpy(
            spectra[train_idx]).float().unsqueeze(-1)  # (N, n_bands, 1)
        train_y = torch.from_numpy(lbls[train_idx]).long()
        test_x = torch.from_numpy(
            spectra[test_idx]).float().unsqueeze(-1)
        test_y = torch.from_numpy(lbls[test_idx]).long()

    else:  # patchwise
        patches, lbls, train_idx, test_idx = _prepare_patchwise_data(
            cube, labels, patch_size=patch_size, band_patch=band_patch)

        model = ViT(
            num_patches=n_bands, num_classes=n_classes, **config
        ).to(_device)

        # Apply gain_neighborhood_band to convert patches
        # (N, patch, patch, n_bands) -> (N, n_bands, patch*patch*band_patch)
        all_patches_t = torch.from_numpy(patches).float()
        all_gnb = gain_neighborhood_band(all_patches_t, band_patch)

        train_x = all_gnb[train_idx]
        train_y = torch.from_numpy(lbls[train_idx]).long()
        test_x = all_gnb[test_idx]
        test_y = torch.from_numpy(lbls[test_idx]).long()

    # DataLoaders
    train_ds = TensorDataset(train_x, train_y)
    test_ds = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Optimizer and scheduler (matching demo.py)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_epochs // 10, gamma=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    tic = time.time()
    best_acc = 0.0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(_device), y_batch.to(_device)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += x_batch.size(0)

        scheduler.step()
        train_acc = correct / total

        # Evaluate
        if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(_device)
                    y_batch = y_batch.to(_device)
                    logits = model(x_batch)
                    test_correct += (
                        logits.argmax(1) == y_batch).sum().item()
                    test_total += x_batch.size(0)
            test_acc = test_correct / test_total
            best_acc = max(best_acc, test_acc)
            print(f"epoch {epoch+1}/{n_epochs}  "
                  f"loss={running_loss/total:.4f}  "
                  f"train_acc={train_acc:.4f}  "
                  f"test_acc={test_acc:.4f}")

    train_time = time.time() - tic

    return model, {
        "variant": variant,
        "n_bands": n_bands,
        "n_classes": n_classes,
        "n_epochs": n_epochs,
        "best_test_acc": best_acc,
        "train_time_sec": train_time,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
