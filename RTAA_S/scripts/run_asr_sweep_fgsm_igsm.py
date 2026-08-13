"""Replicated ASR check vs. FGSM and I-FGSM (IGSM/BIM) baselines — same
protocol as `run_asr_sweep_replicated.py` (RTAA-mismatch vs. PGD) and
`run_asr_sweep_ssfgsm.py` (RTAA-mismatch vs. SS-FGSM).

FGSM (Goodfellow et al., "Explaining and Harnessing Adversarial Examples") is
`rtaa.attacks.baselines.fgsm_attack`: single-step sign-gradient, magnitude
epsilon.

I-FGSM / IGSM / BIM (Kurakin et al., "Adversarial Examples in the Physical
World") is `rtaa.attacks.baselines.pgd_attack` with `random_start=False`:
that flag is the entire difference between PGD and I-FGSM — multi-step
sign-gradient from a deterministic (zero) start, each step clipped to the
epsilon ball, no random initialization. Using the same function as the PGD
baseline (rather than a separate reimplementation) guarantees the step
size/step count/clipping behavior is identical, isolating random-start as the
only varying factor between this baseline and the earlier PGD one.

Only eps=0.01 is used, matching the established protocol: eps>=0.03 was
already shown to be a ceiling effect (near-0% accuracy for every method) in
the original single-seed sweep.
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

from rtaa.attacks.baselines import fgsm_attack, pgd_attack
from rtaa.attacks.spectralformer_attack import (
    PhysicalViabilityWeights,
    SpectralFormerRTAAAttack,
)
from rtaa.models.spectralformer import load_spectralformer_vit
from rtaa.rtm.mismatch import AtmosphericMismatchConfig
from rtaa.rtm.surrogate import RTMSurrogate

EPSILON = 0.01
N_SEEDS = 8
BASELINE_NAMES = ["fgsm", "igsm"]


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

    adv_fgsm = fgsm_attack(
        classifier, clean_spectra.unsqueeze(-1), labels, epsilon=EPSILON,
    ).squeeze(-1)

    adv_igsm = pgd_attack(
        classifier, clean_spectra.unsqueeze(-1), labels,
        epsilon=EPSILON, step_size=EPSILON / 5, n_steps=N_STEPS, random_start=False,
    ).squeeze(-1)

    methods = {"mismatch": adv_mismatch, "fgsm": adv_fgsm, "igsm": adv_igsm}
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
            m = result[f"mismatch__{cond_name}"]
            f = result[f"fgsm__{cond_name}"]
            i = result[f"igsm__{cond_name}"]
            print(f"  {cond_name:15s} ASR: mismatch={m:.4f}  fgsm={f:.4f}  igsm={i:.4f}  "
                  f"(mismatch-fgsm={m-f:+.4f}, mismatch-igsm={m-i:+.4f})")

    with open("asr_sweep_fgsm_igsm_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    n_conditions = len(EVAL_CONDITIONS)
    bonferroni_alpha = 0.05 / (n_conditions * len(BASELINE_NAMES))
    print(f"\n=== Summary across {N_SEEDS} seeds (epsilon={EPSILON}), Bonferroni alpha={bonferroni_alpha:.5f} "
          f"(corrected for {n_conditions} conditions x {len(BASELINE_NAMES)} baselines) ===")
    for baseline_name in BASELINE_NAMES:
        print(f"\n--- RTAA (mismatch-aware) vs. {baseline_name.upper()} ---")
        for cond_name in EVAL_CONDITIONS:
            mismatch_vals = np.array([r[f"mismatch__{cond_name}"] for r in all_results])
            baseline_vals = np.array([r[f"{baseline_name}__{cond_name}"] for r in all_results])
            diff = mismatch_vals - baseline_vals

            _t_stat, p_value = stats.ttest_rel(mismatch_vals, baseline_vals)
            sig = "***" if p_value < bonferroni_alpha else ("*" if p_value < 0.05 else "")

            print(f"\n{cond_name}:")
            print(f"  RTAA (mismatch-aware): {mismatch_vals.mean():.4f} +- {mismatch_vals.std(ddof=1):.4f}")
            print(f"  {baseline_name.upper():22s}: {baseline_vals.mean():.4f} +- {baseline_vals.std(ddof=1):.4f}")
            print(f"  mismatch - {baseline_name}: {diff.mean():+.4f} +- {diff.std(ddof=1):.4f}  (paired t-test p={p_value:.5f}) {sig}")

    print("\nSaved per-seed results to asr_sweep_fgsm_igsm_results.json")


if __name__ == "__main__":
    main()
