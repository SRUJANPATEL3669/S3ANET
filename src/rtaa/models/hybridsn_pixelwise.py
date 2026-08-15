"""HybridSN-Pixelwise: 1D spectral CNN for pixel-level HSI classification.

A pixel-wise variant inspired by the spectral processing branch of HybridSN
(Roy et al., IEEE GRSL 2020). Instead of using 3D-2D spatial-spectral patches,
this model processes individual pixel spectra through 1D convolutions along the
spectral dimension, followed by fully connected layers. This provides a
spatial-context-free baseline to compare against the full patchwise HybridSN.

Input: (B, 1, n_bands) — single pixel spectrum with channel dim.
Output: (B, n_classes) — class logits.
"""

from __future__ import annotations

from torch import Tensor, nn


class HybridSNPixelwise(nn.Module):
    """1D CNN for pixel-level HSI classification without spatial context."""

    def __init__(self, n_bands: int, n_classes: int):
        super().__init__()
        self.n_bands = n_bands

        # 1D spectral convolutions (analogous to HybridSN's spectral processing)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.BatchNorm1d(8),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
        )

        # Pool to reduce spectral dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, 1, n_bands) -> logits (B, n_classes)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(-1)  # (B, 32)
        return self.classifier(x)
