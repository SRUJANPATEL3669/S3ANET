"""RTAA attack against SACNet (Tier 2+ target): a whole-scene fully
convolutional network, not patch- or pixel-wise. Input is the entire
(n_bands, H, W) image at once, so this attack perturbs the whole scene's
reflectance under a single shared atmospheric state (one atmosphere per
captured scene, physically the right granularity) rather than per-pixel/
per-patch atmospheres like the other attack modules.

Uses CrossEntropy2d-style masked loss (matching SACNet's own training code)
so the untargeted attack loss is computed only over a chosen set of labeled
pixels (e.g. the held-out test set), ignoring background/unlabeled/train
pixels — same masking convention as the upstream repo's own FGSM baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state
from rtaa.rtm.surrogate import RTMSurrogate

IGNORE_LABEL = 255


@dataclass
class PhysicalViabilityWeights:
    non_negativity: float = 1.0
    spectral_smoothness: float = 0.1


class SACNetRTAAAttack:
    def __init__(
        self,
        surrogate: RTMSurrogate,
        solar_irradiance: Tensor,
        epsilon: float = 0.05,
        step_size: float = 0.01,
        n_steps: int = 20,
        momentum: float = 0.9,
        phys_weights: PhysicalViabilityWeights | None = None,
        mismatch_config: AtmosphericMismatchConfig | None = None,
        random_start: bool = False,
    ):
        """mismatch_config: see rtaa.rtm.mismatch — without it the
        compensation round-trip is an exact identity (PUBLICATION_ROADMAP.md
        blocking issue). Pass AtmosphericMismatchConfig.none() for the
        RTM-contribution ablation.
        random_start: see `rtaa_attack.RTAAAttack` — initialize delta
        uniformly in the epsilon-ball rather than at 0, matching PGD's
        initialization. Defaults to False to preserve the behavior
        previously-reported results were generated under."""
        self.surrogate = surrogate
        self.solar_irradiance = solar_irradiance
        self.epsilon = epsilon
        self.step_size = step_size
        self.n_steps = n_steps
        self.momentum = momentum
        self.phys_weights = phys_weights or PhysicalViabilityWeights()
        self.mismatch_config = mismatch_config or AtmosphericMismatchConfig()
        self.random_start = random_start

    def _physical_viability_loss(self, r_adv: Tensor) -> Tensor:
        w = self.phys_weights
        non_neg = torch.relu(-r_adv).pow(2).mean()
        smoothness = (r_adv[:, 2:] - 2 * r_adv[:, 1:-1] + r_adv[:, :-2]).pow(2).mean()
        return w.non_negativity * non_neg + w.spectral_smoothness * smoothness

    def generate(
        self,
        classifier: nn.Module,
        clean_scene: Tensor,
        eval_labels: Tensor,
        atm_state: Tensor,
    ) -> tuple[Tensor, dict[str, list[float]]]:
        """clean_scene: (n_bands, H, W), per-band-normalized to [0,1].
        eval_labels: (H, W) long, IGNORE_LABEL (255) at pixels to exclude
        from the attack's loss (background/train/etc — attack only targets
        the pixels you leave un-ignored).
        atm_state: (1, 3), single shared atmosphere for the whole scene.
        Returns (adversarial scene, per-step logs)."""
        if self.random_start:
            delta = torch.empty_like(clean_scene).uniform_(-self.epsilon, self.epsilon).requires_grad_(True)
        else:
            delta = torch.zeros_like(clean_scene, requires_grad=True)
        velocity = torch.zeros_like(clean_scene)

        logs: dict[str, list[float]] = {"adv_loss": [], "phys_loss": [], "total_loss": []}

        atm_state_assumed = perturb_atm_state(atm_state, self.mismatch_config)
        t_atm_true, l_path_true = self.surrogate(atm_state)  # (1, n_bands)
        t_atm_assumed, l_path_assumed = self.surrogate(atm_state_assumed)
        t_atm_true_b = t_atm_true.squeeze(0)[:, None, None]  # (n_bands, 1, 1)
        l_path_true_b = l_path_true.squeeze(0)[:, None, None]
        t_atm_assumed_b = t_atm_assumed.squeeze(0)[:, None, None]
        l_path_assumed_b = l_path_assumed.squeeze(0)[:, None, None]
        solar_b = self.solar_irradiance[:, None, None]

        eval_labels_batched = eval_labels.unsqueeze(0)  # (1, H, W)

        for _ in range(self.n_steps):
            r_adv = torch.clamp(clean_scene + delta, 0.0, 1.0)

            l_sensor = sensor_radiance(r_adv, t_atm_true_b, solar_b, l_path_true_b)
            r_rec = invert_to_reflectance(l_sensor, t_atm_assumed_b, solar_b, l_path_assumed_b)

            logits = classifier(r_rec.unsqueeze(0))  # (1, n_classes, H, W)
            adv_loss = self._masked_cross_entropy(logits, eval_labels_batched)
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

        return torch.clamp(clean_scene + delta, 0.0, 1.0).detach(), logs

    @staticmethod
    def _masked_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
        """logits: (1, C, H, W), labels: (1, H, W) with IGNORE_LABEL entries excluded."""
        n, c, h, w = logits.shape
        mask = labels != IGNORE_LABEL
        logits_flat = logits.permute(0, 2, 3, 1)[mask.view(n, h, w)].view(-1, c)
        labels_flat = labels[mask]
        return nn.functional.cross_entropy(logits_flat, labels_flat)
