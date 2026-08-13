"""Replicated ASR check at epsilon=0.01 — PUBLICATION_ROADMAP.md §1 follow-up.

The single-seed run (`run_asr_sweep.py`, `asr_sweep_results.json`) found a
modest ASR advantage for RTAA (mismatch-aware) over both the ablation and
plain PGD at eps=0.01 in 3/4 conditions, reversing in the 4th. Not
statistically established from one run. This repeats that specific
comparison across multiple seeds (varying test-pixel sample, mismatch noise
draws, and solar irradiance together) to get mean+-std and a paired
significance test on the one comparison that matters: RTAA (mismatch-aware)
vs. RTAA (ablation).

Only eps=0.01 is replicated — eps>=0.03 was already shown to be a ceiling
effect in the single-seed run (0-2% accuracy for every method), so repeating
those wastes compute without adding information.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from run_asr_sweep import (
    EVAL_CONDITIONS,
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
N_SEEDS = 8


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

    rtaa_ablation = SpectralFormerRTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig.none(),
    )
    adv_ablation, _ = rtaa_ablation.generate(classifier=classifier, clean_spectra=clean_spectra, labels=labels, atm_state=gen_atm_state)

    adv_pgd = pgd_attack(
        classifier, clean_spectra.unsqueeze(-1), labels, epsilon=EPSILON, step_size=EPSILON / 5, n_steps=N_STEPS,
    ).squeeze(-1)

    methods = {"mismatch": adv_mismatch, "ablation": adv_ablation, "pgd": adv_pgd}
    seed_result = {"seed": seed}
    for method_name, adv_spectra in methods.items():
        for cond_name, cond_values in EVAL_CONDITIONS.items():
            eval_atm_state = torch.tensor([cond_values] * clean_spectra.shape[0], device=device)
            adv_acc = simulate_and_evaluate(adv_spectra, labels, eval_atm_state, surrogate, solar, classifier)
            seed_result[f"{method_name}__{cond_name}"] = 1.0 - adv_acc  # ASR
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
            m, a, p = result[f"mismatch__{cond_name}"], result[f"ablation__{cond_name}"], result[f"pgd__{cond_name}"]
            print(f"  {cond_name:15s} ASR: mismatch={m:.4f}  ablation={a:.4f}  pgd={p:.4f}  (mismatch-ablation={m-a:+.4f})")

    with open("asr_sweep_replicated_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Summary across {N_SEEDS} seeds (epsilon={EPSILON}) ===")
    for cond_name in EVAL_CONDITIONS:
        mismatch_vals = np.array([r[f"mismatch__{cond_name}"] for r in all_results])
        ablation_vals = np.array([r[f"ablation__{cond_name}"] for r in all_results])
        pgd_vals = np.array([r[f"pgd__{cond_name}"] for r in all_results])
        diff = mismatch_vals - ablation_vals

        _t_stat, p_value = stats.ttest_rel(mismatch_vals, ablation_vals)

        print(f"\n{cond_name}:")
        print(f"  RTAA (mismatch-aware): {mismatch_vals.mean():.4f} +- {mismatch_vals.std(ddof=1):.4f}")
        print(f"  RTAA (ablation):       {ablation_vals.mean():.4f} +- {ablation_vals.std(ddof=1):.4f}")
        print(f"  PGD (baseline):        {pgd_vals.mean():.4f} +- {pgd_vals.std(ddof=1):.4f}")
        print(f"  mismatch - ablation:   {diff.mean():+.4f} +- {diff.std(ddof=1):.4f}  (paired t-test p={p_value:.4f})")

    print("\nSaved per-seed results to asr_sweep_replicated_results.json")


if __name__ == "__main__":
    main()
