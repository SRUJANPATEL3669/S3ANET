"""Differentiable RTM surrogate (H_2 §2.2, feasibility §4).

Predicts atmospheric transmittance T_atm(lambda) and path radiance L_path(lambda)
from scene atmospheric state (aerosol optical depth, water vapor, solar zenith
angle). This is the RTAA analogue of sRTMnet (github.com/pgbrodrick/sRTMnet):
a fast neural emulator standing in for 6S/MODTRAN so gradients can propagate
through the atmosphere back to delta_mat.

Pretrained sRTMnet weights are not vendored here — `RTMSurrogate.from_pretrained`
is the integration point once those weights (or a project-trained equivalent) are
available. Until then, the module trains from simulated (state -> T_atm, L_path)
pairs produced by `rtaa.data.rtm_simulation`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

ATM_STATE_DIM = 3  # (aerosol_optical_depth, water_vapor, solar_zenith_angle)


class RTMSurrogate(nn.Module):
    def __init__(self, n_bands: int, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.n_bands = n_bands

        layers: list[nn.Module] = [nn.Linear(ATM_STATE_DIM, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        self.trunk = nn.Sequential(*layers)

        # Two heads: transmittance in [0, 1], path radiance >= 0.
        self.transmittance_head = nn.Linear(hidden_dim, n_bands)
        self.path_radiance_head = nn.Linear(hidden_dim, n_bands)

    def forward(self, atm_state: Tensor) -> tuple[Tensor, Tensor]:
        """atm_state: (..., 3) -> (T_atm, L_path), each (..., n_bands)."""
        h = self.trunk(atm_state)
        t_atm = torch.sigmoid(self.transmittance_head(h))
        l_path = torch.nn.functional.softplus(self.path_radiance_head(h))
        return t_atm, l_path

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, n_bands: int, **kwargs) -> RTMSurrogate:
        model = cls(n_bands=n_bands, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model
