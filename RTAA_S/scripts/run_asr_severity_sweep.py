"""Single-axis atmospheric severity sweep — PUBLICATION_ROADMAP.md follow-up
to the (corrected, narrowed) 4-condition ASR result.

The 4-condition sweep (`run_asr_sweep_replicated.py`) varied tau, water
vapor, AND solar zenith simultaneously across its conditions, so "distance
from the generation condition" was never a single well-defined quantity —
and after Bonferroni correction, only 2 of 4 conditions were distinguishable
from noise, with a non-monotonic pattern that didn't fit a simple
"generalizes near, fails far" story.

This sweep isolates ONE axis at a time (--axis tau|water_vapor|solar_zenith),
holding the other two fixed at the generation-condition values. The first
sweep point is always the generation value itself, i.e. "distance from
generation" = 0 there. Each axis's range stays within the RTM surrogate's
training range (see `placeholder_physics.py`: tau in [0.05, 0.5], water_vapor
in [0.5, 5.0], solar_zenith in [0, 70]) so extrapolation artifacts (already
ruled out for the tau sweep specifically, but worth avoiding on principle)
aren't a confound on any axis.

Statistical approach: rather than N separate pairwise significance tests
(which needs correction and gets underpowered fast), fit a linear regression
of (RTAA-mismatch ASR minus PGD ASR) against distance-from-generation along
the swept axis, across all (seed, point) pairs, and test whether the slope
is significantly different from zero. This directly tests the actual
hypothesis of interest — does the advantage shrink with distance on this
axis — in one omnibus test instead of many uncorrected ones.
"""

from __future__ import annotations

import argparse
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

# atm_state columns: [tau, water_vapor, solar_zenith]. Each axis's sweep
# starts at the generation value and stays within the surrogate's training
# range (tau in [0.05,0.5], water_vapor in [0.5,5.0], solar_zenith in [0,70]).
AXIS_CONFIG = {
    "tau": {"index": 0, "sweep": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]},
    "water_vapor": {"index": 1, "sweep": [float(v) for v in np.linspace(GENERATION_ATM[1], 5.0, 10)]},
    "solar_zenith": {"index": 2, "sweep": [float(v) for v in np.linspace(GENERATION_ATM[2], 70.0, 10)]},
}


def run_one_seed(
    seed: int, axis_index: int, sweep: list[float],
    device: torch.device, surrogate: RTMSurrogate, classifier: torch.nn.Module,
) -> dict:
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
    for value in sweep:
        atm_values = list(GENERATION_ATM)
        atm_values[axis_index] = float(value)
        eval_atm = torch.tensor([atm_values] * clean_spectra.shape[0], device=device, dtype=torch.float32)
        acc_mismatch = simulate_and_evaluate(adv_mismatch, labels, eval_atm, surrogate, solar, classifier)
        acc_pgd = simulate_and_evaluate(adv_pgd, labels, eval_atm, surrogate, solar, classifier)
        seed_result[f"mismatch__{value:.3f}"] = 1.0 - acc_mismatch  # ASR
        seed_result[f"pgd__{value:.3f}"] = 1.0 - acc_pgd
    return seed_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=list(AXIS_CONFIG), default="tau")
    args = parser.parse_args()

    axis_index = AXIS_CONFIG[args.axis]["index"]
    sweep = AXIS_CONFIG[args.axis]["sweep"]
    generation_value = GENERATION_ATM[axis_index]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier = load_spectralformer_vit("Indian", device=device)
    classifier.eval()
    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=200).to(device)
    surrogate.eval()

    all_results = []
    for seed in range(N_SEEDS):
        print(f"=== axis={args.axis} seed {seed}/{N_SEEDS-1} ===")
        result = run_one_seed(seed, axis_index, sweep, device, surrogate, classifier)
        all_results.append(result)

    out_path = f"asr_severity_sweep_{args.axis}_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Per-point summary across {N_SEEDS} seeds, axis={args.axis} (epsilon={EPSILON}) ===")
    distances, diffs_for_regression = [], []
    for value in sweep:
        mismatch_vals = np.array([r[f"mismatch__{value:.3f}"] for r in all_results])
        pgd_vals = np.array([r[f"pgd__{value:.3f}"] for r in all_results])
        diff = mismatch_vals - pgd_vals
        distance = value - generation_value

        _t_stat, p_value = stats.ttest_rel(mismatch_vals, pgd_vals)
        print(f"{args.axis}={value:.3f} (dist={distance:+.3f})  mismatch={mismatch_vals.mean():.4f}±{mismatch_vals.std(ddof=1):.4f}  "
              f"pgd={pgd_vals.mean():.4f}±{pgd_vals.std(ddof=1):.4f}  diff={diff.mean():+.4f}  p={p_value:.4f}")

        distances.extend([distance] * N_SEEDS)
        diffs_for_regression.extend(diff.tolist())

    slope, intercept, r_value, p_slope, _std_err = stats.linregress(distances, diffs_for_regression)
    n_conditions = len(sweep)
    bonferroni_alpha = 0.05 / n_conditions

    print(f"\n=== Trend test: does (mismatch - PGD) ASR advantage change with distance along {args.axis}? ===")
    print(f"slope={slope:+.5f} per unit distance, intercept={intercept:+.4f}, r^2={r_value**2:.4f}")
    print(f"p-value for slope != 0: {p_slope:.5f}")
    print(f"Bonferroni alpha for {n_conditions} per-point tests would be {bonferroni_alpha:.4f} (reference only)")
    if p_slope < 0.05:
        direction = "shrinks" if slope < 0 else "grows"
        print(f"=> Advantage {direction} significantly with distance along {args.axis} (p<0.05).")
    else:
        print(f"=> No significant linear trend detected along {args.axis}.")

    print(f"\nSaved per-seed results to {out_path}")


if __name__ == "__main__":
    main()
