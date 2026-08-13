"""RTAA attack generator (H_2 §2.2-2.3): Momentum-PGD over physical material
perturbations, propagated through the differentiable RTM surrogate and a
differentiable compensation step to the classifier.

Tier-1 simplification (see project roadmap Phase 2): delta_mat is parametrized
as a bounded additive perturbation directly on surface reflectance, rather than
the full thin-film/refractive-index optical model in H_2 §2.3. This is enough
to validate that gradients flow correctly end-to-end through the physical
chain; the full optical-coating parametrization is a Tier-2 extension once the
pipeline is validated on HybridSN.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state
from rtaa.rtm.surrogate import RTMSurrogate


class DifferentiablePCA(nn.Module):
    """Differentiable re-implementation of a fitted sklearn PCA, for attacking
    raw reflectance spectra while feeding PCA-space patches to the classifier."""

    def __init__(self, mean: Tensor, components: Tensor, whiten_scale: Tensor | None = None):
        super().__init__()
        self.register_buffer("mean", mean)  # (n_bands,)
        self.register_buffer("components", components)  # (n_components, n_bands)
        if whiten_scale is not None:
            self.register_buffer("whiten_scale", whiten_scale)  # (n_components,)
        else:
            self.whiten_scale = None

    def forward(self, spectra: Tensor) -> Tensor:
        """spectra: (..., n_bands) -> (..., n_components)."""
        centered = spectra - self.mean
        projected = centered @ self.components.T
        if self.whiten_scale is not None:
            projected = projected / self.whiten_scale
        return projected


@dataclass
class PhysicalViabilityWeights:
    non_negativity: float = 1.0
    spectral_smoothness: float = 0.1
    bound_penalty: float = 1.0


class RTAAAttack:
    """Momentum-PGD attack through delta_mat -> RTM surrogate -> compensation -> classifier."""

    def __init__(
        self,
        surrogate: RTMSurrogate,
        solar_irradiance: Tensor,
        epsilon: float = 0.05,
        step_size: float = 0.01,
        n_steps: int = 20,
        momentum: float = 0.9,
        adv_loss_weight: float = 1.0,
        phys_weights: PhysicalViabilityWeights | None = None,
        mismatch_config: AtmosphericMismatchConfig | None = None,
        random_start: bool = False,
        grad_normalization: str = "mean",
    ):
        """mismatch_config controls the gap between the atmosphere used to
        generate the scene and the atmosphere assumed during compensation
        (PUBLICATION_ROADMAP.md, blocking issue) — without it the RTM
        round-trip is an exact identity and this reduces to a plain
        reflectance-domain attack. Pass `AtmosphericMismatchConfig.none()` to
        recover that identity behavior (the RTM-contribution ablation).

        random_start: initialize delta ~ U(-epsilon, epsilon) instead of 0.
        This is the one structural difference between this attack and
        `baselines.pgd_attack` that was never ablated — momentum, gradient
        normalization, and the physical-viability loss were each tested and
        each turned out to *help* (PUBLICATION_ROADMAP.md), leaving random
        initialization as the remaining candidate explanation for the
        scaffolding cost (ablation underperforming PGD) measured on 5/6
        target architectures. Defaults to False to preserve the behavior all
        previously-reported results were generated under.

        grad_normalization: "mean" (default, preserves all previously-reported
        results) divides the raw gradient by its mean absolute value each
        step, matching the original implementation. "median" divides by the
        median absolute value instead — a robust statistic, motivated
        directly by the gradient-concentration diagnostic
        (PUBLICATION_ROADMAP.md): on architectures where a handful of
        outlier dimensions carry ~all the gradient mass (HybridSN,
        concentration 0.999), a mean-based normalizer is dominated by those
        outliers and momentum accumulates whatever direction they happen to
        point in, rather than the many small, informative dimensions the
        mean-based scheme effectively down-weights."""
        self.surrogate = surrogate
        self.solar_irradiance = solar_irradiance
        self.epsilon = epsilon
        self.step_size = step_size
        self.n_steps = n_steps
        self.momentum = momentum
        self.adv_loss_weight = adv_loss_weight
        self.phys_weights = phys_weights or PhysicalViabilityWeights()
        self.mismatch_config = mismatch_config or AtmosphericMismatchConfig()
        self.random_start = random_start
        if grad_normalization not in ("mean", "median"):
            raise ValueError(f"grad_normalization must be 'mean' or 'median', got {grad_normalization!r}")
        self.grad_normalization = grad_normalization

    def _physical_viability_loss(self, r_adv: Tensor) -> Tensor:
        w = self.phys_weights
        non_neg = torch.relu(-r_adv).pow(2).mean()
        smoothness = (r_adv[..., 2:] - 2 * r_adv[..., 1:-1] + r_adv[..., :-2]).pow(2).mean()
        return w.non_negativity * non_neg + w.spectral_smoothness * smoothness

    def generate(
        self,
        classifier: nn.Module,
        pca_projector: DifferentiablePCA,
        clean_spectra: Tensor,
        clean_patch_for_shape: Tensor,
        labels: Tensor,
        atm_state: Tensor,
    ) -> tuple[Tensor, dict[str, list[float]]]:
        """Runs the attack and returns (adversarial reflectance spectra, per-step logs).

        clean_spectra: (B, patch, patch, n_bands) raw surface reflectance patches.
        clean_patch_for_shape: (B, 1, n_components, patch, patch) shape reference,
            i.e. the PCA'd patch as the classifier normally consumes it.
        labels: (B,) ground-truth class indices.
        atm_state: (B, 3) or (1, 3) TRUE atmospheric state the scene was
            captured under. The compensation step uses a perturbed version of
            this (see AtmosphericMismatchConfig), not the same value.
        """
        if self.random_start:
            delta = torch.empty_like(clean_spectra).uniform_(-self.epsilon, self.epsilon).requires_grad_(True)
        else:
            delta = torch.zeros_like(clean_spectra, requires_grad=True)
        velocity = torch.zeros_like(clean_spectra)
        loss_fn = nn.CrossEntropyLoss()

        logs: dict[str, list[float]] = {"adv_loss": [], "phys_loss": [], "total_loss": []}

        atm_state_assumed = perturb_atm_state(atm_state, self.mismatch_config)
        t_atm_true, l_path_true = self.surrogate(atm_state)  # (B_or_1, n_bands)
        t_atm_assumed, l_path_assumed = self.surrogate(atm_state_assumed)

        for _ in range(self.n_steps):
            r_adv = torch.clamp(clean_spectra + delta, 0.0, 1.0)

            l_sensor = sensor_radiance(r_adv, t_atm_true[:, None, None, :], self.solar_irradiance, l_path_true[:, None, None, :])
            r_rec = invert_to_reflectance(l_sensor, t_atm_assumed[:, None, None, :], self.solar_irradiance, l_path_assumed[:, None, None, :])

            # (B, patch, patch, n_components) -> (B, 1, n_components, patch, patch)
            pca_patch = pca_projector(r_rec).permute(0, 3, 1, 2).unsqueeze(1)

            logits = classifier(pca_patch)
            adv_loss = loss_fn(logits, labels)
            phys_loss = self._physical_viability_loss(r_adv)
            total_loss = self.adv_loss_weight * adv_loss - phys_loss

            grad = torch.autograd.grad(total_loss, delta, retain_graph=False)[0]
            if self.grad_normalization == "median":
                scale = grad.abs().median()
            else:
                scale = grad.abs().mean()
            grad = grad / (scale + 1e-12)
            velocity = self.momentum * velocity + grad
            delta = delta.detach() + self.step_size * velocity.sign()
            delta = torch.clamp(delta, -self.epsilon, self.epsilon).requires_grad_(True)

            logs["adv_loss"].append(adv_loss.item())
            logs["phys_loss"].append(phys_loss.item())
            logs["total_loss"].append(total_loss.item())

        return torch.clamp(clean_spectra + delta, 0.0, 1.0).detach(), logs
