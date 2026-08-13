"""Replicated ASR check vs. Carlini-Wagner L2 — same protocol as the other
baseline comparisons, with one structural difference flagged explicitly:
C&W (`rtaa.attacks.baselines.cw_attack`) is not epsilon-bounded like
FGSM/PGD/IGSM/SS-FGSM/RTAA. It minimizes perturbation L2 norm subject to a
misclassification constraint (trading off via `c`), so there is no shared
epsilon to hold fixed across methods here. Reporting ASR alone would let C&W
"win" for free simply by using a larger perturbation budget than RTAA's
epsilon=0.01 — so this script also reports the resulting mean L_inf
perturbation magnitude for both methods alongside ASR, so the comparison's
budget asymmetry (if any) is visible rather than hidden.

Found and fixed a real bug in `cw_attack` while wiring this up: its box
constraint mapped adversarial reflectance into (-1, 1) via a vanilla tanh
reparametrization, instead of (0, 1) — the actual valid reflectance domain
used everywhere else in this codebase. Fixed to the standard [0,1]-domain
C&W box constraint (`adv = 0.5*(tanh(w)+1)`, `w = atanh(2*original-1)`)
before running this comparison.

CW_C is calibrated (scripts/calibrate_cw_budget.py) so C&W's mean L_inf
perturbation matches RTAA's fixed epsilon=0.01 exactly (0.0100 vs 0.0100).
An earlier run at the uncalibrated default c=10 used 3.4x RTAA's budget
(mean L_inf 0.0343) and produced an uninformative ASR "win" for C&W that
was entirely explained by the larger budget, not attack quality — see
PUBLICATION_ROADMAP.md. This run is the corrected, budget-matched version.

Only eps=0.01 is used for RTAA, matching the established protocol.
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

from rtaa.attacks.baselines import cw_attack
from rtaa.attacks.spectralformer_attack import (
    PhysicalViabilityWeights,
    SpectralFormerRTAAAttack,
)
from rtaa.models.spectralformer import load_spectralformer_vit
from rtaa.rtm.mismatch import AtmosphericMismatchConfig
from rtaa.rtm.surrogate import RTMSurrogate

EPSILON = 0.01
N_SEEDS = 8
CW_STEPS = 200
CW_LR = 0.01
CW_C = 0.00125  # calibrated via scripts/calibrate_cw_budget.py to match RTAA's mean L_inf=0.01


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

    adv_cw = cw_attack(
        classifier, clean_spectra.unsqueeze(-1), labels, n_steps=CW_STEPS, lr=CW_LR, c=CW_C,
    ).squeeze(-1)

    linf_mismatch = (adv_mismatch - clean_spectra).abs().max(dim=1).values.mean().item()
    linf_cw = (adv_cw - clean_spectra).abs().max(dim=1).values.mean().item()

    methods = {"mismatch": adv_mismatch, "cw": adv_cw}
    seed_result = {"seed": seed, "linf_mismatch": linf_mismatch, "linf_cw": linf_cw}
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
        print(f"  mean L_inf perturbation: mismatch={result['linf_mismatch']:.4f}  cw={result['linf_cw']:.4f}")
        for cond_name in EVAL_CONDITIONS:
            m, c = result[f"mismatch__{cond_name}"], result[f"cw__{cond_name}"]
            print(f"  {cond_name:15s} ASR: mismatch={m:.4f}  cw={c:.4f}  (mismatch-cw={m-c:+.4f})")

    with open("asr_sweep_cw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    linf_mismatch_vals = np.array([r["linf_mismatch"] for r in all_results])
    linf_cw_vals = np.array([r["linf_cw"] for r in all_results])
    n_conditions = len(EVAL_CONDITIONS)
    bonferroni_alpha = 0.05 / n_conditions

    print(f"\n=== Perturbation budget check (mean L_inf per sample, across {N_SEEDS} seeds) ===")
    print(f"  RTAA (mismatch-aware), epsilon={EPSILON}: {linf_mismatch_vals.mean():.4f} +- {linf_mismatch_vals.std(ddof=1):.4f}")
    print(f"  C&W (c={CW_C}, unconstrained):            {linf_cw_vals.mean():.4f} +- {linf_cw_vals.std(ddof=1):.4f}")
    if linf_cw_vals.mean() > linf_mismatch_vals.mean():
        ratio = linf_cw_vals.mean() / linf_mismatch_vals.mean()
        print(f"  => C&W's perturbation is {ratio:.1f}x LARGER on average — any ASR advantage for C&W is expected and not a fair win.")

    print(f"\n=== ASR summary across {N_SEEDS} seeds, Bonferroni alpha={bonferroni_alpha:.4f} ===")
    for cond_name in EVAL_CONDITIONS:
        mismatch_vals = np.array([r[f"mismatch__{cond_name}"] for r in all_results])
        cw_vals = np.array([r[f"cw__{cond_name}"] for r in all_results])
        diff = mismatch_vals - cw_vals

        _t_stat, p_value = stats.ttest_rel(mismatch_vals, cw_vals)
        sig = "***" if p_value < bonferroni_alpha else ("*" if p_value < 0.05 else "")

        print(f"\n{cond_name}:")
        print(f"  RTAA (mismatch-aware): {mismatch_vals.mean():.4f} +- {mismatch_vals.std(ddof=1):.4f}")
        print(f"  C&W:                   {cw_vals.mean():.4f} +- {cw_vals.std(ddof=1):.4f}")
        print(f"  mismatch - cw:         {diff.mean():+.4f} +- {diff.std(ddof=1):.4f}  (paired t-test p={p_value:.5f}) {sig}")

    print("\nSaved per-seed results to asr_sweep_cw_results.json")


if __name__ == "__main__":
    main()
