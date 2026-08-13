"""HybridSN: 3D-2D CNN for HSI classification (Tier 1 cheap network).

Used first to validate the RTAA attack pipeline cheaply before scaling to
ViT / Mamba-based defenses (Tier 2), per the two-tier validation strategy in
the RTAA project roadmap. Reference: Roy et al., "HybridSN: Exploring 3-D-2-D
CNN Feature Hierarchy for Hyperspectral Image Classification," IEEE GRSL 2020.
"""

from __future__ import annotations

from torch import Tensor, nn


class HybridSN(nn.Module):
    def __init__(
        self,
        n_bands: int,
        n_classes: int,
        patch_size: int = 25,
        pca_components: int = 30,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pca_components = pca_components

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3)), nn.ReLU(inplace=True)
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3)), nn.ReLU(inplace=True)
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)), nn.ReLU(inplace=True)
        )

        spectral_depth_after_3d = pca_components - (7 - 1) - (5 - 1) - (3 - 1)
        conv2d_in_channels = 32 * spectral_depth_after_3d

        self.conv2d = nn.Sequential(
            nn.Conv2d(conv2d_in_channels, 64, kernel_size=(3, 3)), nn.ReLU(inplace=True)
        )

        spatial_after = patch_size - 2 * 3 - 2  # three 3D convs (-2 each on H,W) + one 2D conv (-2)
        flat_dim = 64 * spatial_after * spatial_after

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, 1, pca_components, patch_size, patch_size) -> logits (B, n_classes)."""
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)

        b, c, d, h, w = x.shape
        x = x.reshape(b, c * d, h, w)
        x = self.conv2d(x)

        x = x.flatten(start_dim=1)
        return self.classifier(x)
