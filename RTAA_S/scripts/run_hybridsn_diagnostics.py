"""Two follow-ups to the HybridSN ASR result (PUBLICATION_ROADMAP.md §1):
RTAA (mismatch-aware) loses to plain PGD in all 4 conditions at eps=0.01,
despite beating its own ablation — the opposite of the SpectralFormer
pattern. Two hypotheses were flagged for that, both tested here in one run
(shares data loading/sampling across both checks per seed):

(A) Ceiling-effect hypothesis: both methods already hit 92-98% ASR at
    eps=0.01 (vs. 60-72% on SpectralFormer at the same eps) — HybridSN may
    just be far more vulnerable on this patch representation, compressing
    the room for any method's advantage to show up. Tests this by sweeping
    epsilon down to well below saturation (0.002-0.01) and checking whether
    a real win/loss pattern (rather than a compressed one) appears at lower
    budget, mirroring how the eps=0.01 operating point was originally found
    for SpectralFormer.

(B) Physical-viability-cost hypothesis: `RTAAAttack`'s physical-viability
    loss (non-negativity + spectral-smoothness penalty, subtracted from the
    adversarial loss every step) is a real constraint PGD doesn't pay, and
    may be the actual reason for the loss rather than anything about
    atmosphere-awareness specifically. Tests this by rerunning RTAA
    (mismatch-aware) at eps=0.01 with `phys_weights` zeroed out entirely and
    comparing directly to the already-measured PGD numbers from the same
    seeds (`asr_sweep_hybridsn_results.json`) — if the gap to PGD closes or
    reverses with phys_weights=0, that confirms the physical-viability loss
    (not the mismatch-awareness mechanism) is what's costing ASR.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from run_asr_sweep_hybridsn import (
    CHECKPOINT,
    EVAL_CONDITIONS,
    GENERATION_ATM,
    N_SAMPLES,
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

from rtaa.attacks.baselines import pgd_attack
from rtaa.attacks.rtaa_attack import PhysicalViabilityWeights, RTAAAttack
from rtaa.rtm.mismatch import AtmosphericMismatchConfig

N_SEEDS = 8
EPSILON_SWEEP = [0.002, 0.004, 0.006, 0.008, 0.01]  # eps=0.01 doubles as the phys_weights=0 comparison point
ZERO_PHYS_WEIGHTS = PhysicalViabilityWeights(non_negativity=0.0, spectral_smoothness=0.0)


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

    seed_result: dict = {"seed": seed}

    for epsilon in EPSILON_SWEEP:
        rtaa_mismatch = RTAAAttack(
            surrogate=surrogate, solar_irradiance=solar, epsilon=epsilon, step_size=epsilon / 5,
            n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig(),
        )
        adv_mismatch, _ = rtaa_mismatch.generate(
            classifier=hybridsn, pca_projector=pca_projector, clean_spectra=clean_patches,
            clean_patch_for_shape=clean_patches, labels=labels, atm_state=gen_atm_state,
        )
        adv_pgd = pgd_attack(
            wrapped_classifier, clean_patches, labels, epsilon=epsilon, step_size=epsilon / 5, n_steps=N_STEPS,
        )

        variants = {"mismatch": adv_mismatch, "pgd": adv_pgd}

        if epsilon == 0.01:
            rtaa_nophys = RTAAAttack(
                surrogate=surrogate, solar_irradiance=solar, epsilon=epsilon, step_size=epsilon / 5,
                n_steps=N_STEPS, phys_weights=ZERO_PHYS_WEIGHTS, mismatch_config=AtmosphericMismatchConfig(),
            )
            adv_nophys, _ = rtaa_nophys.generate(
                classifier=hybridsn, pca_projector=pca_projector, clean_spectra=clean_patches,
                clean_patch_for_shape=clean_patches, labels=labels, atm_state=gen_atm_state,
            )
            variants["mismatch_nophys"] = adv_nophys

        for variant_name, adv_patches in variants.items():
            for cond_name, cond_values in EVAL_CONDITIONS.items():
                eval_atm_state = torch.tensor([cond_values] * clean_patches.shape[0], device=device)
                adv_acc = simulate_and_evaluate_patches(adv_patches, labels, eval_atm_state, surrogate, solar, wrapped_classifier)
                seed_result[f"{variant_name}__eps{epsilon}__{cond_name}"] = 1.0 - adv_acc

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

    all_results = []
    for seed in range(N_SEEDS):
        print(f"=== seed {seed}/{N_SEEDS-1} ===")
        result = run_one_seed(seed, test_rows, test_cols, test_labels, cube_norm, device, surrogate, pca_projector, hybridsn, wrapped_classifier)
        all_results.append(result)
        for epsilon in EPSILON_SWEEP:
            m = result[f"mismatch__eps{epsilon}__clear"]
            p = result[f"pgd__eps{epsilon}__clear"]
            print(f"  eps={epsilon:.3f}  clear: mismatch={m:.4f}  pgd={p:.4f}  diff={m-p:+.4f}")

    with open("hybridsn_diagnostics_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== (A) Ceiling-effect check: mismatch vs. PGD across epsilon, all conditions ===")
    n_eps = len(EPSILON_SWEEP)
    n_conditions = len(EVAL_CONDITIONS)
    bonferroni_alpha_a = 0.05 / (n_eps * n_conditions)
    for epsilon in EPSILON_SWEEP:
        print(f"\n--- epsilon={epsilon} ---")
        for cond_name in EVAL_CONDITIONS:
            m = np.array([r[f"mismatch__eps{epsilon}__{cond_name}"] for r in all_results])
            p = np.array([r[f"pgd__eps{epsilon}__{cond_name}"] for r in all_results])
            diff = m - p
            _t, pval = stats.ttest_rel(m, p)
            sig = "***" if pval < bonferroni_alpha_a else ("*" if pval < 0.05 else "")
            print(f"  {cond_name:15s} mismatch={m.mean():.4f}±{m.std(ddof=1):.4f}  pgd={p.mean():.4f}±{p.std(ddof=1):.4f}  diff={diff.mean():+.4f}  p={pval:.5f} {sig}")

    print("\n=== (B) Physical-viability-cost check at epsilon=0.01: mismatch (phys_weights=0) vs. PGD ===")
    bonferroni_alpha_b = 0.05 / n_conditions
    for cond_name in EVAL_CONDITIONS:
        nophys = np.array([r[f"mismatch_nophys__eps0.01__{cond_name}"] for r in all_results])
        p = np.array([r[f"pgd__eps0.01__{cond_name}"] for r in all_results])
        default_m = np.array([r[f"mismatch__eps0.01__{cond_name}"] for r in all_results])
        diff = nophys - p
        _t, pval = stats.ttest_rel(nophys, p)
        sig = "***" if pval < bonferroni_alpha_b else ("*" if pval < 0.05 else "")
        print(f"\n{cond_name}:")
        print(f"  RTAA (mismatch, default phys_weights): {default_m.mean():.4f}")
        print(f"  RTAA (mismatch, phys_weights=0):       {nophys.mean():.4f}±{nophys.std(ddof=1):.4f}")
        print(f"  PGD:                                   {p.mean():.4f}±{p.std(ddof=1):.4f}")
        print(f"  (nophys - pgd): {diff.mean():+.4f}  p={pval:.5f} {sig}")

    print("\nSaved per-seed results to hybridsn_diagnostics_results.json")


if __name__ == "__main__":
    main()
