"""Domain-randomization follow-up to the effective-tau failure law
(PUBLICATION_ROADMAP.md/PAPER_ROADMAP.md §4 item 2 — the highest-priority
new experiment identified for the paper).

RTAA as validated everywhere else generates an attack under a single fixed
"true" generation atmosphere (clear: tau=0.05, water_vapor=0.8,
zenith=15deg) and a correspondingly narrow retrieval-error gap around it.
The factorial-corner experiment showed this attack loses to plain PGD
specifically when the *evaluation-time* true atmosphere pushes the
effective slant-path optical depth (tau_eff = tau*(sec(zenith)+1)) above
~1.0-1.5 — a condition the attacker never saw or accounted for during
generation.

This script tests whether generating under a WIDE distribution of possible
true atmospheres, rather than one fixed "clear" condition, recovers
robustness at the optically-thick corner without losing the wins elsewhere.
At each optimization step, K=4 independent true-atmosphere realizations are
sampled uniformly across the full range used throughout this project's
severity sweeps (tau in [0.05,0.5], water_vapor in [0.8,4.0], zenith in
[15,60]), each with its own independently-sampled retrieval-error gap
(AtmosphericMismatchConfig), and the classification loss is averaged across
all K before the gradient step — a standard expectation-over-transformation
(EOT) recipe, applied here to atmospheric uncertainty rather than a
generic image transformation.

Compares three attacks head-to-head on SpectralFormer (pixel-wise)/Indian
Pines, epsilon=0.01, 8 seeds:
  - RTAA (mismatch, fixed clear generation) — the original, already-tested config
  - RTAA (domain-randomized, K=4 wide true-atmosphere sampling) — new
  - PGD — reference baseline, unaffected by any of this

Evaluated at the 4 standard conditions AND, critically, at the two
factorial corners where the original RTAA lost to PGD (HLH: tau=0.5,
water_vapor=0.8, zenith=60; HHH: tau=0.5, water_vapor=4.0, zenith=60).
"""

from __future__ import annotations

import json

import numpy as np
import torch
from run_asr_sweep import load_data, simulate_and_evaluate
from scipy import stats
from torch import Tensor, nn

from rtaa.attacks.baselines import pgd_attack
from rtaa.attacks.spectralformer_attack import (
    PhysicalViabilityWeights,
    SpectralFormerRTAAAttack,
)
from rtaa.models.spectralformer import load_spectralformer_vit
from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state
from rtaa.rtm.surrogate import RTMSurrogate

RTM_CHECKPOINT = "checkpoints/rtm_surrogate_200bands.pt"
GENERATION_ATM = [0.05, 0.8, 15.0]
EVAL_CONDITIONS = {
    "clear": [0.05, 0.8, 15.0],
    "moderate_haze": [0.2, 1.5, 30.0],
    "heavy_haze": [0.4, 2.5, 45.0],
    "extreme": [0.5, 4.0, 60.0],
    "factorial_HLH": [0.5, 0.8, 60.0],   # a losing corner in the original factorial experiment
    "factorial_HHH": [0.5, 4.0, 60.0],   # the other losing corner
}
EPSILON = 0.01
N_SAMPLES = 300
N_STEPS = 25
N_SEEDS = 8
K_ENSEMBLE = 4

WIDE_ATM_RANGES = {
    "tau": (0.05, 0.5),
    "water_vapor": (0.8, 4.0),
    "solar_zenith": (15.0, 60.0),
}


def sample_wide_atm_state(batch_size: int, device: torch.device, generator: torch.Generator) -> Tensor:
    tau = torch.empty(batch_size, 1).uniform_(*WIDE_ATM_RANGES["tau"], generator=generator)
    wv = torch.empty(batch_size, 1).uniform_(*WIDE_ATM_RANGES["water_vapor"], generator=generator)
    zenith = torch.empty(batch_size, 1).uniform_(*WIDE_ATM_RANGES["solar_zenith"], generator=generator)
    return torch.cat([tau, wv, zenith], dim=1).to(device)


def generate_domain_randomized(
    classifier: nn.Module, clean_spectra: Tensor, labels: Tensor,
    surrogate: RTMSurrogate, solar: Tensor, mismatch_config: AtmosphericMismatchConfig,
    epsilon: float, step_size: float, n_steps: int, momentum: float,
    phys_weights: PhysicalViabilityWeights, k_ensemble: int, seed: int,
) -> Tensor:
    device = clean_spectra.device
    batch_size = clean_spectra.shape[0]
    generator = torch.Generator().manual_seed(seed + 10_000)  # offset from other seeded draws; CPU, for sample_wide_atm_state
    device_generator = torch.Generator(device=device).manual_seed(seed + 20_000)  # for perturb_atm_state's on-device noise

    delta = torch.zeros_like(clean_spectra, requires_grad=True)
    velocity = torch.zeros_like(clean_spectra)
    loss_fn = nn.CrossEntropyLoss()

    def phys_loss_fn(r_adv: Tensor) -> Tensor:
        non_neg = torch.relu(-r_adv).pow(2).mean()
        smoothness = (r_adv[..., 2:] - 2 * r_adv[..., 1:-1] + r_adv[..., :-2]).pow(2).mean()
        return phys_weights.non_negativity * non_neg + phys_weights.spectral_smoothness * smoothness

    for _ in range(n_steps):
        r_adv = torch.clamp(clean_spectra + delta, 0.0, 1.0)
        total_loss = torch.zeros((), device=device)

        for _ in range(k_ensemble):
            atm_state_true_k = sample_wide_atm_state(batch_size, device, generator)
            atm_state_assumed_k = perturb_atm_state(atm_state_true_k, mismatch_config, generator=device_generator)
            t_atm_true, l_path_true = surrogate(atm_state_true_k)
            t_atm_assumed, l_path_assumed = surrogate(atm_state_assumed_k)

            l_sensor = sensor_radiance(r_adv, t_atm_true, solar, l_path_true)
            r_rec = invert_to_reflectance(l_sensor, t_atm_assumed, solar, l_path_assumed)

            logits = classifier(r_rec.unsqueeze(-1))
            total_loss = total_loss + loss_fn(logits, labels)

        adv_loss = total_loss / k_ensemble
        phys_loss = phys_loss_fn(r_adv)
        combined_loss = adv_loss - phys_loss

        grad = torch.autograd.grad(combined_loss, delta, retain_graph=False)[0]
        grad = grad / (grad.abs().mean() + 1e-12)
        velocity = momentum * velocity + grad
        delta = delta.detach() + step_size * velocity.sign()
        delta = torch.clamp(delta, -epsilon, epsilon).requires_grad_(True)

    return torch.clamp(clean_spectra + delta, 0.0, 1.0).detach()


def run_one_seed(seed: int, device: torch.device, surrogate: RTMSurrogate, classifier: nn.Module) -> dict:
    torch.manual_seed(seed)
    clean_spectra, labels, n_bands = load_data(device, N_SAMPLES, seed=seed)
    solar = (torch.rand(n_bands, generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.75).to(device)
    gen_atm_state = torch.tensor([GENERATION_ATM] * clean_spectra.shape[0], device=device)

    rtaa_fixed = SpectralFormerRTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig(),
    )
    adv_fixed, _ = rtaa_fixed.generate(classifier=classifier, clean_spectra=clean_spectra, labels=labels, atm_state=gen_atm_state)

    adv_domain_rand = generate_domain_randomized(
        classifier=classifier, clean_spectra=clean_spectra, labels=labels,
        surrogate=surrogate, solar=solar, mismatch_config=AtmosphericMismatchConfig(),
        epsilon=EPSILON, step_size=EPSILON / 5, n_steps=N_STEPS, momentum=0.9,
        phys_weights=PhysicalViabilityWeights(), k_ensemble=K_ENSEMBLE, seed=seed,
    )

    adv_pgd = pgd_attack(classifier, clean_spectra.unsqueeze(-1), labels, epsilon=EPSILON, step_size=EPSILON / 5, n_steps=N_STEPS).squeeze(-1)

    methods = {"fixed": adv_fixed, "domain_rand": adv_domain_rand, "pgd": adv_pgd}
    seed_result = {"seed": seed}
    for method_name, adv_spectra in methods.items():
        for cond_name, cond_values in EVAL_CONDITIONS.items():
            eval_atm_state = torch.tensor([cond_values] * clean_spectra.shape[0], device=device)
            adv_acc = simulate_and_evaluate(adv_spectra, labels, eval_atm_state, surrogate, solar, classifier)
            seed_result[f"{method_name}__{cond_name}"] = 1.0 - adv_acc
    return seed_result


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier = load_spectralformer_vit("Indian", device=device)
    classifier.eval()
    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=200).to(device)
    surrogate.eval()

    all_results = []
    for seed in range(N_SEEDS):
        print(f"=== seed {seed}/{N_SEEDS-1} ===")
        result = run_one_seed(seed, device, surrogate, classifier)
        all_results.append(result)
        for cond_name in EVAL_CONDITIONS:
            f, d, p = result[f"fixed__{cond_name}"], result[f"domain_rand__{cond_name}"], result[f"pgd__{cond_name}"]
            print(f"  {cond_name:16s} ASR: fixed={f:.4f}  domain_rand={d:.4f}  pgd={p:.4f}  (domain_rand-pgd={d-p:+.4f})")

    with open("domain_randomization_defense_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    n_conditions = len(EVAL_CONDITIONS)
    bonferroni_alpha = 0.05 / n_conditions
    print(f"\n=== Summary across {N_SEEDS} seeds (epsilon={EPSILON}), Bonferroni alpha={bonferroni_alpha:.4f} ===")
    for label_a, label_b in [("domain_rand", "pgd"), ("domain_rand", "fixed"), ("fixed", "pgd")]:
        print(f"\n--- {label_a} vs. {label_b} ---")
        for cond_name in EVAL_CONDITIONS:
            a_vals = np.array([r[f"{label_a}__{cond_name}"] for r in all_results])
            b_vals = np.array([r[f"{label_b}__{cond_name}"] for r in all_results])
            diff = a_vals - b_vals
            _t, p_value = stats.ttest_rel(a_vals, b_vals)
            sig = "***" if p_value < bonferroni_alpha else ("*" if p_value < 0.05 else "")
            print(f"{cond_name:16s} {label_a}={a_vals.mean():.4f}  {label_b}={b_vals.mean():.4f}  diff={diff.mean():+.4f}  p={p_value:.5f} {sig}")

    print("\nSaved per-seed results to domain_randomization_defense_results.json")


if __name__ == "__main__":
    main()
