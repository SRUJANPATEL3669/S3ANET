"""Third follow-up to the HybridSN ASR result: tests the optimizer-mismatch
hypothesis flagged after both other hypotheses (ceiling effect, physical-
viability cost) were refuted by `run_hybridsn_diagnostics.py`.

`RTAAAttack`'s update rule is momentum (0.9) + gradient normalization
(`grad / grad.abs().mean()`) each step, validated against SpectralFormer's
compact 200-dim per-pixel attack surface. With momentum=0, this reduces
exactly to plain iterative sign-gradient ascent from a zero start: velocity
becomes just `grad` each step (no accumulation), and dividing by the positive
scalar `grad.abs().mean()` before taking `.sign()` has literally no effect on
the sign — so `delta += step_size * velocity.sign()` becomes
`delta += step_size * grad.sign()`, which is exactly PGD's/I-FGSM's update
rule. This makes momentum=0 a clean, principled isolation of "RTAA's
optimizer, stripped to the same update rule as PGD" while keeping everything
else (the RTM forward/compensation round-trip, the physical-viability loss)
unchanged.

If the gap to PGD closes with momentum=0, that confirms the momentum/
normalization scheme (tuned for a much smaller attack surface) is the actual
driver of RTAA's HybridSN underperformance, not anything about the physics
or the physical-viability loss (both already ruled out).

Only epsilon=0.01 is tested — the established operating point, and (per the
epsilon sweep in run_hybridsn_diagnostics.py) the point where the gap to PGD
was already smallest, so the cleanest place to check whether it closes
further or reverses. Reuses the exact same per-seed sampling as
run_asr_sweep_hybridsn.py (same seeded RNG calls in the same order), so the
already-saved PGD numbers in asr_sweep_hybridsn_results.json are directly
comparable without rerunning PGD.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from run_asr_sweep_hybridsn import (
    CHECKPOINT,
    EPSILON,
    EVAL_CONDITIONS,
    GENERATION_ATM,
    N_SAMPLES,
    N_SEEDS,
    N_STEPS,
    PATCH_SIZE,
    PCA_COMPONENTS,
    RTM_CHECKPOINT,
    SPLIT_SEED,
    TRAIN_FRACTION,
    HybridSN,
    RTMSurrogate,
    _WrappedClassifier,
    build_pca_projector,
    extract_raw_patches,
    load_hsi_cube,
    normalize_reflectance,
    simulate_and_evaluate_patches,
    stratified_train_test_split,
)
from scipy import stats

from rtaa.attacks.rtaa_attack import PhysicalViabilityWeights, RTAAAttack
from rtaa.rtm.mismatch import AtmosphericMismatchConfig


def run_one_seed(
    seed: int, test_rows: np.ndarray, test_cols: np.ndarray, test_labels: np.ndarray,
    cube_norm: np.ndarray, device: torch.device, surrogate: RTMSurrogate,
    pca_projector, hybridsn: HybridSN, wrapped_classifier,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(test_rows), min(N_SAMPLES, len(test_rows)), replace=False)
    rows, cols = test_rows[sel], test_cols[sel]
    labels_np = test_labels[sel] - 1

    raw_patches_np = extract_raw_patches(cube_norm, rows, cols, PATCH_SIZE)
    clean_patches = torch.from_numpy(raw_patches_np).to(device)
    labels = torch.from_numpy(labels_np).long().to(device)

    n_bands = cube_norm.shape[-1]
    solar = (torch.rand(n_bands, generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.75).to(device)
    gen_atm_state = torch.tensor([GENERATION_ATM] * clean_patches.shape[0], device=device)

    rtaa_nomomentum = RTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, momentum=0.0, phys_weights=PhysicalViabilityWeights(),
        mismatch_config=AtmosphericMismatchConfig(),
    )
    adv_nomomentum, _ = rtaa_nomomentum.generate(
        classifier=hybridsn, pca_projector=pca_projector, clean_spectra=clean_patches,
        clean_patch_for_shape=clean_patches, labels=labels, atm_state=gen_atm_state,
    )

    seed_result = {"seed": seed}
    for cond_name, cond_values in EVAL_CONDITIONS.items():
        eval_atm_state = torch.tensor([cond_values] * clean_patches.shape[0], device=device)
        adv_acc = simulate_and_evaluate_patches(adv_nomomentum, labels, eval_atm_state, surrogate, solar, wrapped_classifier)
        seed_result[f"nomomentum__{cond_name}"] = 1.0 - adv_acc
    return seed_result


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cube, labels_map = load_hsi_cube("IndianPines")
    cube_norm = normalize_reflectance(cube)

    rows_all, cols_all = np.nonzero(labels_map != 0)
    entry_labels = labels_map[rows_all, cols_all].tolist()
    _train_idx, test_idx = stratified_train_test_split(entry_labels, TRAIN_FRACTION, SPLIT_SEED)
    test_rows, test_cols = rows_all[test_idx], cols_all[test_idx]
    test_labels = labels_map[test_rows, test_cols]

    pca_projector = build_pca_projector(cube_norm, PCA_COMPONENTS, device)

    with open(CHECKPOINT.replace(".pt", ".json")) as f:
        meta = json.load(f)
    hybridsn = HybridSN(
        n_bands=meta["n_bands"], n_classes=meta["n_classes"],
        patch_size=meta["patch_size"], pca_components=meta["pca_components"],
    ).to(device)
    hybridsn.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    hybridsn.eval()

    wrapped_classifier = _WrappedClassifier(pca_projector, hybridsn).to(device)
    wrapped_classifier.eval()

    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=200).to(device)
    surrogate.eval()

    with open("asr_sweep_hybridsn_results.json") as f:
        prior_results = json.load(f)

    all_results = []
    for seed in range(N_SEEDS):
        print(f"=== seed {seed}/{N_SEEDS-1} ===")
        result = run_one_seed(seed, test_rows, test_cols, test_labels, cube_norm, device, surrogate, pca_projector, hybridsn, wrapped_classifier)
        all_results.append(result)
        prior = prior_results[seed]
        assert prior["seed"] == seed
        for cond_name in EVAL_CONDITIONS:
            nm = result[f"nomomentum__{cond_name}"]
            p = prior[f"pgd__{cond_name}"]
            m = prior[f"mismatch__{cond_name}"]
            print(f"  {cond_name:15s} nomomentum={nm:.4f}  pgd={p:.4f}  (orig mismatch={m:.4f})  nomomentum-pgd={nm-p:+.4f}")

    with open("hybridsn_momentum_check_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    n_conditions = len(EVAL_CONDITIONS)
    bonferroni_alpha = 0.05 / n_conditions
    print(f"\n=== RTAA (momentum=0, mismatch-aware) vs. PGD, epsilon={EPSILON}, Bonferroni alpha={bonferroni_alpha:.4f} ===")
    for cond_name in EVAL_CONDITIONS:
        nomomentum_vals = np.array([r[f"nomomentum__{cond_name}"] for r in all_results])
        pgd_vals = np.array([prior_results[i][f"pgd__{cond_name}"] for i in range(N_SEEDS)])
        orig_mismatch_vals = np.array([prior_results[i][f"mismatch__{cond_name}"] for i in range(N_SEEDS)])
        diff = nomomentum_vals - pgd_vals

        _t_stat, p_value = stats.ttest_rel(nomomentum_vals, pgd_vals)
        sig = "***" if p_value < bonferroni_alpha else ("*" if p_value < 0.05 else "")

        print(f"\n{cond_name}:")
        print(f"  RTAA (momentum=0.9, original): {orig_mismatch_vals.mean():.4f}")
        print(f"  RTAA (momentum=0):             {nomomentum_vals.mean():.4f} +- {nomomentum_vals.std(ddof=1):.4f}")
        print(f"  PGD:                           {pgd_vals.mean():.4f} +- {pgd_vals.std(ddof=1):.4f}")
        print(f"  (momentum=0) - pgd:            {diff.mean():+.4f} +- {diff.std(ddof=1):.4f}  (paired t-test p={p_value:.5f}) {sig}")

    print("\nSaved per-seed results to hybridsn_momentum_check_results.json")


if __name__ == "__main__":
    main()
