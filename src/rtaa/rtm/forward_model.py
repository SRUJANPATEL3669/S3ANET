"""Differentiable forward radiative-transfer model (H_2 §2.1).

L_sensor(lambda) = T_atm(lambda) * R_surface(lambda) * E_sun(lambda) + L_path(lambda)
"""

from __future__ import annotations

import torch
from torch import Tensor


def sensor_radiance(
    surface_reflectance: Tensor,
    atm_transmittance: Tensor,
    solar_irradiance: Tensor,
    path_radiance: Tensor,
) -> Tensor:
    """Forward-model at-sensor radiance from surface reflectance and atmospheric state.

    All tensors are broadcastable along a trailing spectral-band dimension, e.g.
    shape (..., n_bands). Fully differentiable w.r.t. every argument so gradients
    can flow back to `surface_reflectance` (and, through it, to delta_mat).
    """
    return atm_transmittance * surface_reflectance * solar_irradiance + path_radiance


def invert_to_reflectance(
    sensor_radiance_: Tensor,
    atm_transmittance: Tensor,
    solar_irradiance: Tensor,
    path_radiance: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Closed-form reflectance recovery given known atmospheric state.

    This is the differentiable stand-in for empirical compensation algorithms
    (FLAASH/QUAC) when the atmospheric state is known/estimated by the RTM
    surrogate rather than retrieved by an opaque solver. See H_2 §4 open risk:
    FLAASH/QUAC themselves are not differentiable, so RTAA either estimates the
    atmospheric state directly (as here) or wraps a learned inverse.
    """
    denom = atm_transmittance * solar_irradiance
    denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
    return (sensor_radiance_ - path_radiance) / denom
