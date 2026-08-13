"""RTAA attack against SpectralFormer (Tier 2 target) — both the plain-ViT
pixel-wise variant and the actual SpectralFormer patch-wise/CAF architecture.

Same physical chain as `rtaa_attack.RTAAAttack` (delta_mat -> RTM surrogate ->
forward radiance -> closed-form compensation -> classifier), but adapted to
SpectralFormer's input format: full per-band-normalized spectrum, no PCA.
Kept as a separate module rather than folded into RTAAAttack because the
input pipeline (no PCA, per-band-not-global normalization, and — for
patch-wise — the group-wise spectral embedding transform) is different
enough that sharing one class would need a pile of conditionals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from rtaa.models.spectralformer import gain_neighborhood_band
from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state
from rtaa.rtm.surrogate import RTMSurrogate


def per_band_normalize(cube_flat: np.ndarray) -> np.ndarray:
    """Per-band min-max normalize a (N, n_bands) array to [0, 1], matching
    SpectralFormer's own training-time normalization (demo.py)."""
    lo = cube_flat.min(axis=0, keepdims=True)
    hi = cube_flat.max(axis=0, keepdims=True)
    return (cube_flat - lo) / (hi - lo + 1e-12)


@dataclass
class PhysicalViabilityWeights:
    non_negativity: float = 1.0
    spectral_smoothness: float = 0.1


class SpectralFormerRTAAAttack:
    def __init__(
        self,
        surrogate: RTMSurrogate,
        solar_irradiance: Tensor,
        epsilon: float = 0.05,
        step_size: float = 0.01,
        n_steps: int = 20,
        momentum: float = 0.9,
        phys_weights: PhysicalViabilityWeights | None = None,
        band_patch: int | None = None,
        mismatch_config: AtmosphericMismatchConfig | None = None,
        random_start: bool = False,
    ):
        """band_patch=None -> pixel-wise variant, clean_spectra is (B, n_bands),
        classifier input built as `r_rec.unsqueeze(-1)`.
        band_patch=int -> patch-wise/CAF variant, clean_spectra is
        (B, patch, patch, n_bands), classifier input built via
        `gain_neighborhood_band` (group-wise spectral embedding).
        mismatch_config: see rtaa.rtm.mismatch — without it the compensation
        round-trip is an exact identity (PUBLICATION_ROADMAP.md blocking
        issue). Pass AtmosphericMismatchConfig.none() for the RTM-ablation.
        random_start: see `rtaa_attack.RTAAAttack` — initialize delta
        uniformly in the epsilon-ball rather than at 0, matching
        `baselines.pgd_attack`. Defaults to False to preserve the behavior
        previously-reported results were generated under."""
        self.surrogate = surrogate
        self.solar_irradiance = solar_irradiance
        self.epsilon = epsilon
        self.step_size = step_size
        self.n_steps = n_steps
        self.momentum = momentum
        self.phys_weights = phys_weights or PhysicalViabilityWeights()
        self.band_patch = band_patch
        self.mismatch_config = mismatch_config or AtmosphericMismatchConfig()
        self.random_start = random_start

    def _physical_viability_loss(self, r_adv: Tensor) -> Tensor:
        w = self.phys_weights
        non_neg = torch.relu(-r_adv).pow(2).mean()
        smoothness = (r_adv[..., 2:] - 2 * r_adv[..., 1:-1] + r_adv[..., :-2]).pow(2).mean()
        return w.non_negativity * non_neg + w.spectral_smoothness * smoothness

    def _build_classifier_input(self, r_rec: Tensor) -> Tensor:
        if self.band_patch is None:
            return r_rec.unsqueeze(-1)  # (B, n_bands) -> (B, n_bands, 1)
        return gain_neighborhood_band(r_rec, self.band_patch)  # (B, p, p, n_bands) -> (B, n_bands, p*p*band_patch)

    def generate(
        self,
        classifier: nn.Module,
        clean_spectra: Tensor,
        labels: Tensor,
        atm_state: Tensor,
    ) -> tuple[Tensor, dict[str, list[float]]]:
        """clean_spectra: (B, n_bands) for pixel-wise, or (B, patch, patch,
        n_bands) for patch-wise — per-band-normalized to [0,1] either way.
        Returns (adversarial spectra, per-step logs)."""
        if self.random_start:
            delta = torch.empty_like(clean_spectra).uniform_(-self.epsilon, self.epsilon).requires_grad_(True)
        else:
            delta = torch.zeros_like(clean_spectra, requires_grad=True)
        velocity = torch.zeros_like(clean_spectra)
        loss_fn = nn.CrossEntropyLoss()

        logs: dict[str, list[float]] = {"adv_loss": [], "phys_loss": [], "total_loss": []}

        atm_state_assumed = perturb_atm_state(atm_state, self.mismatch_config)
        t_atm_true, l_path_true = self.surrogate(atm_state)  # (B, n_bands)
        t_atm_assumed, l_path_assumed = self.surrogate(atm_state_assumed)

        # broadcast atm outputs across any spatial dims in clean_spectra
        extra_dims = clean_spectra.ndim - t_atm_true.ndim
        broadcast_shape = t_atm_true.shape[:1] + (1,) * extra_dims + t_atm_true.shape[1:]
        t_atm_true_b = t_atm_true.reshape(broadcast_shape)
        l_path_true_b = l_path_true.reshape(broadcast_shape)
        t_atm_assumed_b = t_atm_assumed.reshape(broadcast_shape)
        l_path_assumed_b = l_path_assumed.reshape(broadcast_shape)

        for _ in range(self.n_steps):
            r_adv = torch.clamp(clean_spectra + delta, 0.0, 1.0)

            l_sensor = sensor_radiance(r_adv, t_atm_true_b, self.solar_irradiance, l_path_true_b)
            r_rec = invert_to_reflectance(l_sensor, t_atm_assumed_b, self.solar_irradiance, l_path_assumed_b)

            logits = classifier(self._build_classifier_input(r_rec))
            adv_loss = loss_fn(logits, labels)
            phys_loss = self._physical_viability_loss(r_adv)
            total_loss = adv_loss - phys_loss

            grad = torch.autograd.grad(total_loss, delta, retain_graph=False)[0]
            grad = grad / (grad.abs().mean() + 1e-12)
            velocity = self.momentum * velocity + grad
            delta = delta.detach() + self.step_size * velocity.sign()
            delta = torch.clamp(delta, -self.epsilon, self.epsilon).requires_grad_(True)

            logs["adv_loss"].append(adv_loss.item())
            logs["phys_loss"].append(phys_loss.item())
            logs["total_loss"].append(total_loss.item())

        return torch.clamp(clean_spectra + delta, 0.0, 1.0).detach(), logs
