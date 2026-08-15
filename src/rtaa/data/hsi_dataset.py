"""HSI benchmark loading, PCA preprocessing, and patch extraction.

Supports PaviaU, Indian Pines, Houston 2018 (H_2 §1.A) as .mat files with a
data cube key and a ground-truth label key. Datasets are drawn from the local
mirror at `/home/jayant/projects/SAFER/data/` (shared across projects — do not
re-download). File layout there is quirky and was confirmed by inspecting the
actual arrays, not by filename alone:

- PaviaU / Indian Pines: standard scipy-readable v5 .mat, filenames match
  content (`*_gt.mat` holds integer class labels).
- Houston 2018: MATLAB v7.3 (HDF5) files, and the data/label roles are
  **swapped relative to filename convention** — `houston18.mat` holds the
  label map (key "map", 8 classes), while `houston18_gt.mat` holds the
  reflectance cube (key "ori_data", shape (48, H, W), already scaled to
  [0, 1]). This was verified empirically (unique-value counts) since the
  naming is actively misleading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

DEFAULT_DATA_DIR = os.environ.get("RTAA_DATA_DIR", "/content/drive/MyDrive/S3Anet_data")


@dataclass(frozen=True)
class DatasetSpec:
    data_file: str
    data_key: str
    label_file: str
    label_key: str
    loader: str  # "scipy" or "h5py"
    cube_axis_order: str  # "hwc" or "chw"


DATASET_SPECS: dict[str, DatasetSpec] = {
    "PaviaU": DatasetSpec("paviaU.mat", "paviaU", "paviaU_gt.mat", "paviaU_gt", "scipy", "hwc"),
    "IndianPines": DatasetSpec(
        "indian_pines_corrected.mat", "indian_pines_corrected",
        "indian_pines_gt.mat", "indian_pines_gt", "scipy", "hwc",
    ),
    "Houston2018": DatasetSpec(
        "houston18_gt.mat", "ori_data", "houston18.mat", "map", "h5py", "chw",
    ),
    "Salinas": DatasetSpec(
        "Salinas_corrected.mat", "salinas_corrected", "Salinas_gt.mat", "salinas_gt", "scipy", "hwc",
    ),
}


def _load_array(path: Path, key: str, loader: str) -> np.ndarray:
    if loader == "scipy":
        return loadmat(path)[key]
    with h5py.File(path, "r") as f:
        return np.array(f[key])


def load_hsi_cube(dataset_name: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cube, labels): cube (H, W, n_bands) float32, labels (H, W) int64."""
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {dataset_name!r}, expected one of {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[dataset_name]
    data_dir = Path(data_dir)

    cube = _load_array(data_dir / spec.data_file, spec.data_key, spec.loader).astype(np.float32)
    if spec.cube_axis_order == "chw":
        cube = np.transpose(cube, (1, 2, 0))

    labels = _load_array(data_dir / spec.label_file, spec.label_key, spec.loader).astype(np.int64)
    return cube, labels


def normalize_reflectance(cube: np.ndarray) -> np.ndarray:
    """Rescale a raw-DN cube (e.g. PaviaU/Indian Pines values in the thousands)
    to [0, 1] so it's physically meaningful as a reflectance fraction.

    Required before feeding a cube into `rtaa.rtm.forward_model` /
    `rtaa.attacks.rtaa_attack` — those assume reflectance in [0, 1] (the RTAA
    attack hard-clamps to [0, 1] every step; raw DN values in the thousands
    saturate that clamp to a constant, silently killing the attack's gradient
    signal). NOT needed before `HSIPatchDataset`/classifier training — PCA
    with whitening is invariant to a global positive rescale, so classifiers
    trained on raw-DN PCA features are unaffected either way.
    """
    return cube / cube.max()


def apply_pca(cube: np.ndarray, n_components: int) -> np.ndarray:
    """cube (H, W, n_bands) -> (H, W, n_components), band dim reduced via PCA."""
    h, w, n_bands = cube.shape
    flat = cube.reshape(-1, n_bands)
    reduced = PCA(n_components=n_components, whiten=True).fit_transform(flat)
    return reduced.reshape(h, w, n_components)


def pad_cube(cube: np.ndarray, margin: int) -> np.ndarray:
    return np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")


@dataclass
class PatchIndex:
    row: int
    col: int
    label: int


def build_patch_index(labels: np.ndarray, ignore_label: int = 0) -> list[PatchIndex]:
    rows, cols = np.nonzero(labels != ignore_label)
    return [PatchIndex(r, c, int(labels[r, c])) for r, c in zip(rows, cols)]


class HSIPatchDataset(Dataset):
    """Extracts (patch_size, patch_size, pca_components) neighborhoods around labeled pixels."""

    def __init__(
        self,
        cube: np.ndarray,
        labels: np.ndarray,
        patch_size: int = 25,
        pca_components: int = 30,
        ignore_label: int = 0,
    ):
        self.patch_size = patch_size
        self.margin = patch_size // 2
        self.pca_components = pca_components

        pca_cube = apply_pca(cube, pca_components)
        self.padded_cube = pad_cube(pca_cube, self.margin)
        self.index = build_patch_index(labels, ignore_label)
        # Labels are 1-indexed with 0 = unlabeled; shift to 0-indexed classes.
        self.n_classes = int(labels.max())

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.index[idx]
        r, c = entry.row + self.margin, entry.col + self.margin
        m = self.margin
        patch = self.padded_cube[r - m : r + m + 1, c - m : c + m + 1, :]
        # (patch, patch, pca_components) -> (1, pca_components, patch, patch)
        patch = np.transpose(patch, (2, 0, 1))[None, ...]
        return torch.from_numpy(patch.astype(np.float32)), torch.tensor(entry.label - 1)


def make_synthetic_dataset(
    height: int = 40,
    width: int = 40,
    n_bands: int = 103,
    n_classes: int = 9,
    patch_size: int = 25,
    pca_components: int = 30,
    seed: int = 0,
) -> HSIPatchDataset:
    """Synthetic cube for pipeline smoke tests when real .mat files aren't available."""
    rng = np.random.default_rng(seed)
    cube = rng.random((height, width, n_bands), dtype=np.float32)
    labels = rng.integers(0, n_classes + 1, size=(height, width)).astype(np.int64)
    return HSIPatchDataset(cube, labels, patch_size=patch_size, pca_components=pca_components)
