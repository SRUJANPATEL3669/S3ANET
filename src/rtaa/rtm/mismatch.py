"""Atmospheric compensation mismatch (PUBLICATION_ROADMAP.md, blocking issue).

Without this, `sensor_radiance` followed by `invert_to_reflectance` using the
same atmospheric state is an exact mathematical identity — the "attack
through the atmosphere" step does nothing a plain reflectance-domain attack
couldn't. Real atmospheric compensation is never perfect: retrieval
algorithms estimate aerosol optical depth and water vapor with known error
bounds, so the state assumed during compensation differs from the true state
the sensor actually saw. This module models that gap.

Solar zenith angle is geometry (sun position, acquisition time), not a
retrieved quantity — no error is injected there by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# atm_state columns: [aerosol_optical_depth, water_vapor_cm, solar_zenith_deg]
_TAU_COL, _WATER_VAPOR_COL, _SOLAR_ZENITH_COL = 0, 1, 2


@dataclass
class AtmosphericMismatchConfig:
    tau_bias: float = 0.03
    tau_noise_std: float = 0.01
    water_vapor_bias: float = 0.1
    water_vapor_noise_std: float = 0.05
    solar_zenith_bias: float = 0.0
    solar_zenith_noise_std: float = 0.0

    @classmethod
    def none(cls) -> AtmosphericMismatchConfig:
        """Zero mismatch — recovers the old identity-map behavior. Use this
        for the RTM-contribution ablation (PUBLICATION_ROADMAP.md §3): compare
        results with this config against the default to see whether modeling
        atmospheric-compensation error actually changes attack outcomes."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def perturb_atm_state(
    atm_state_true: Tensor,
    config: AtmosphericMismatchConfig | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Returns atm_state_assumed: what a compensation algorithm would
    estimate, given known retrieval error, for the true atmospheric state
    the sensor experienced. Fixed per call (not per-attack-step) — models a
    single compensation-algorithm run's error for one scene, not per-step
    randomness within a single attack."""
    config = config or AtmosphericMismatchConfig()
    bias = torch.tensor(
        [config.tau_bias, config.water_vapor_bias, config.solar_zenith_bias],
        device=atm_state_true.device, dtype=atm_state_true.dtype,
    )
    noise_std = torch.tensor(
        [config.tau_noise_std, config.water_vapor_noise_std, config.solar_zenith_noise_std],
        device=atm_state_true.device, dtype=atm_state_true.dtype,
    )
    noise = torch.randn(atm_state_true.shape, device=atm_state_true.device, generator=generator) * noise_std
    atm_state_assumed = atm_state_true + bias + noise

    # aerosol optical depth and water vapor are physically non-negative
    atm_state_assumed[..., _TAU_COL] = atm_state_assumed[..., _TAU_COL].clamp(min=0.0)
    atm_state_assumed[..., _WATER_VAPOR_COL] = atm_state_assumed[..., _WATER_VAPOR_COL].clamp(min=0.0)
    return atm_state_assumed
