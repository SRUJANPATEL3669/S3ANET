"""Factorial verification of the joint-extreme interaction hypothesis
(PUBLICATION_ROADMAP.md follow-up to the 3 single-axis severity sweeps).

Each individual axis (tau, water_vapor, solar_zenith), swept alone, showed
RTAA (mismatch-aware) robustly beating PGD across its full range with no
decay — yet the original joint sweep found a significant LOSS at a condition
where all three were simultaneously extreme. That's only possible if it's an
interaction effect, not a univariate one. This script pins down exactly
which combination(s) of extreme axes trigger it: a full 2^3 factorial design,
each axis at its generation value ("low") or its extreme value ("high"),
evaluated at all 8 corners of the resulting cube. The (low,low,low) corner
reproduces the generation condition itself (a sanity check against the
single-axis sweeps' zero-distance point); the (high,high,high) corner
reproduces the original "extreme" condition from the first 4-condition sweep
(a sanity check against that earlier result).

Attack generation is identical to the other ASR scripts (same generation
atmosphere, same epsilon, same n_steps) — only the evaluation grid differs,
so this reuses the same 2 attacks (RTAA-mismatch, PGD) per seed and just
evaluates them at 8 points instead of 4 or 10.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import torch
from run_asr_sweep import (
    GENERATION_ATM,
    N_SAMPLES,
    N_STEPS,
    RTM_CHECKPOINT,
    load_data,
    simulate_and_evaluate,
)
from scipy import stats

from rtaa.attacks.baselines import pgd_attack
from rtaa.attacks.spectralformer_attack import (
    PhysicalViabilityWeights,
    SpectralFormerRTAAAttack,
)
from rtaa.models.spectralformer import load_spectralformer_vit
from rtaa.rtm.mismatch import AtmosphericMismatchConfig
from rtaa.rtm.surrogate import RTMSurrogate

EPSILON = 0.01
N_SEEDS = 15

AXIS_NAMES = ["tau", "water_vapor", "solar_zenith"]
LOW_VALUES = GENERATION_ATM  # [0.05, 0.8, 15.0] — the generation condition itself
HIGH_VALUES = [0.5, 4.0, 60.0]  # matches the original "extreme" condition exactly

CORNERS = list(itertools.product([0, 1], repeat=3))  # 8 corners, 0=low 1=high


def corner_label(corner: tuple[int, int, int]) -> str:
    return "".join("H" if bit else "L" for bit in corner)


def corner_atm_state(corner: tuple[int, int, int]) -> list[float]:
    return [HIGH_VALUES[i] if bit else LOW_VALUES[i] for i, bit in enumerate(corner)]


def run_one_seed(seed: int, device: torch.device, surrogate: RTMSurrogate, classifier: torch.nn.Module) -> dict:
    torch.manual_seed(seed)
    clean_spectra, labels, n_bands = load_data(device, N_SAMPLES, seed=seed)
    solar = (torch.rand(n_bands, generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.75).to(device)
    gen_atm_state = torch.tensor([GENERATION_ATM] * clean_spectra.shape[0], device=device)

    rtaa_mismatch = SpectralFormerRTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig(),
    )
    adv_mismatch, _ = rtaa_mismatch.generate(classifier=classifier, clean_spectra=clean_spectra, labels=labels, atm_state=gen_atm_state)

    adv_pgd = pgd_attack(
        classifier, clean_spectra.unsqueeze(-1), labels, epsilon=EPSILON, step_size=EPSILON / 5, n_steps=N_STEPS,
    ).squeeze(-1)

    seed_result = {"seed": seed}
    for corner in CORNERS:
        label = corner_label(corner)
        atm_values = corner_atm_state(corner)
        eval_atm = torch.tensor([atm_values] * clean_spectra.shape[0], device=device, dtype=torch.float32)
        acc_mismatch = simulate_and_evaluate(adv_mismatch, labels, eval_atm, surrogate, solar, classifier)
        acc_pgd = simulate_and_evaluate(adv_pgd, labels, eval_atm, surrogate, solar, classifier)
        seed_result[f"mismatch__{label}"] = 1.0 - acc_mismatch
        seed_result[f"pgd__{label}"] = 1.0 - acc_pgd
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
        all_results.append(run_one_seed(seed, device, surrogate, classifier))

    with open("asr_factorial_corners_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    n_corners = len(CORNERS)
    bonferroni_alpha = 0.05 / n_corners
    print("\n=== Factorial corners (tau, water_vapor, solar_zenith), L=generation value, H=extreme value ===")
    print(f"Bonferroni alpha for {n_corners} corners: {bonferroni_alpha:.5f}\n")
    print(f"{'corner':8s} {'tau':>6s} {'wv':>6s} {'zenith':>7s}  {'mismatch':>16s} {'pgd':>16s} {'diff':>9s} {'p-value':>9s}  significant?")
    for corner in CORNERS:
        label = corner_label(corner)
        atm_values = corner_atm_state(corner)
        mismatch_vals = np.array([r[f"mismatch__{label}"] for r in all_results])
        pgd_vals = np.array([r[f"pgd__{label}"] for r in all_results])
        diff = mismatch_vals - pgd_vals
        _t, p = stats.ttest_rel(mismatch_vals, pgd_vals)
        sig = "***" if p < bonferroni_alpha else ("worse" if diff.mean() < 0 and p < 0.05 else "")
        print(f"{label:8s} {atm_values[0]:6.2f} {atm_values[1]:6.2f} {atm_values[2]:7.1f}  "
              f"{mismatch_vals.mean():7.4f}±{mismatch_vals.std(ddof=1):.4f} {pgd_vals.mean():7.4f}±{pgd_vals.std(ddof=1):.4f} "
              f"{diff.mean():+.4f}  {p:9.5f}  {sig}")

    print("\nSaved per-seed results to asr_factorial_corners_results.json")


if __name__ == "__main__":
    main()
