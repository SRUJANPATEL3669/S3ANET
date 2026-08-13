"""Fourth hypothesis for the RTAA-vs-PGD scaffolding cost: random start.

The scaffolding decomposition (PUBLICATION_ROADMAP.md) showed that RTAA's
losses/nulls vs. PGD are driven entirely by `(ablation - PGD)` being
negative on 5/6 architectures -- not by the atmosphere-mismatch mechanism,
whose contribution `(mismatch - ablation)` is positive on 6/6 targets and 23
of 24 condition x architecture cells. On HybridSN the scaffolding cost is
-10.9pp, swamping a +7.0pp mechanism gain.

Three explanations were already tested and each REFUTED, all in the same
direction -- each component turned out to be helping, not hurting:
  (A) ceiling effect        -> gap widens at lower epsilon
  (B) physical-viability loss -> removing it makes RTAA worse
  (C) momentum/grad-normalization -> removing it makes RTAA worse

All three asked "is something RTAA *adds* hurting it?". This script asks the
inverse, never tested: "is something PGD *has* that RTAA lacks helping PGD?"
There is exactly one such structural difference remaining --
`baselines.pgd_attack` initializes delta ~ U(-eps, eps) while all three RTAA
attack classes initialize delta = 0. Random initialization is precisely what
distinguishes PGD from I-FGSM, and the earlier I-FGSM comparison already
showed that difference is measurable on this pipeline.

Tests RTAA (mismatch-aware) and RTAA (ablation) with random_start=True
against the already-measured PGD numbers from the same seeds
(asr_sweep_hybridsn_results.json), on HybridSN -- the worst-affected target,
so the clearest signal. Reports the scaffolding decomposition before and
after, so the question "did random start close the scaffolding gap?" is
answered directly rather than inferred from the net number.
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


def run_one_seed(seed, test_rows, test_cols, test_labels, cube_norm, device, surrogate, pca_projector, hybridsn, wrapped_classifier):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(test_rows), min(N_SAMPLES, len(test_rows)), replace=False)
    rows, cols = test_rows[sel], test_cols[sel]
    labels_np = test_labels[sel] - 1

    clean_patches = torch.from_numpy(extract_raw_patches(cube_norm, rows, cols, PATCH_SIZE)).to(device)
    labels = torch.from_numpy(labels_np).long().to(device)
    n_bands = cube_norm.shape[-1]
    solar = (torch.rand(n_bands, generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.75).to(device)
    gen_atm_state = torch.tensor([GENERATION_ATM] * clean_patches.shape[0], device=device)

    variants = {}
    for name, mismatch_cfg in [("mismatch_rs", AtmosphericMismatchConfig()), ("ablation_rs", AtmosphericMismatchConfig.none())]:
        attack = RTAAAttack(
            surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
            n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=mismatch_cfg,
            random_start=True,
        )
        adv, _ = attack.generate(
            classifier=hybridsn, pca_projector=pca_projector, clean_spectra=clean_patches,
            clean_patch_for_shape=clean_patches, labels=labels, atm_state=gen_atm_state,
        )
        variants[name] = adv

    seed_result = {"seed": seed}
    for name, adv in variants.items():
        for cond_name, cond_values in EVAL_CONDITIONS.items():
            eval_atm = torch.tensor([cond_values] * clean_patches.shape[0], device=device)
            acc = simulate_and_evaluate_patches(adv, labels, eval_atm, surrogate, solar, wrapped_classifier)
            seed_result[f"{name}__{cond_name}"] = 1.0 - acc
    return seed_result


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cube, labels_map = load_hsi_cube("IndianPines")
    cube_norm = normalize_reflectance(cube)
    rows_all, cols_all = np.nonzero(labels_map != 0)
    entry_labels = labels_map[rows_all, cols_all].tolist()
    _tr, test_idx = stratified_train_test_split(entry_labels, TRAIN_FRACTION, SPLIT_SEED)
    test_rows, test_cols = rows_all[test_idx], cols_all[test_idx]
    test_labels = labels_map[test_rows, test_cols]

    pca_projector = build_pca_projector(cube_norm, PCA_COMPONENTS, device)
    with open(CHECKPOINT.replace(".pt", ".json")) as f:
        meta = json.load(f)
    hybridsn = HybridSN(n_bands=meta["n_bands"], n_classes=meta["n_classes"],
                        patch_size=meta["patch_size"], pca_components=meta["pca_components"]).to(device)
    hybridsn.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    hybridsn.eval()
    wrapped = _WrappedClassifier(pca_projector, hybridsn).to(device)
    wrapped.eval()
    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=200).to(device)
    surrogate.eval()

    with open("asr_sweep_hybridsn_results.json") as f:
        prior = json.load(f)

    results = []
    for seed in range(N_SEEDS):
        print(f"=== seed {seed}/{N_SEEDS-1} ===")
        r = run_one_seed(seed, test_rows, test_cols, test_labels, cube_norm, device, surrogate, pca_projector, hybridsn, wrapped)
        results.append(r)
        for c in EVAL_CONDITIONS:
            print(f"  {c:15s} mismatch_rs={r[f'mismatch_rs__{c}']:.4f}  ablation_rs={r[f'ablation_rs__{c}']:.4f}  "
                  f"(prior no-RS mismatch={prior[seed][f'mismatch__{c}']:.4f}, pgd={prior[seed][f'pgd__{c}']:.4f})")

    with open("hybridsn_random_start_results.json", "w") as f:
        json.dump(results, f, indent=2)

    alpha = 0.05 / len(EVAL_CONDITIONS)
    print(f"\n=== HybridSN scaffolding decomposition, before vs. after random start (Bonferroni alpha={alpha:.4f}) ===")
    print(f"{'condition':15s} {'scaffold OLD':>13s} {'scaffold NEW':>13s} {'net OLD':>9s} {'net NEW':>9s} {'net NEW p':>11s}")
    for c in EVAL_CONDITIONS:
        pgd = np.array([prior[i][f"pgd__{c}"] for i in range(N_SEEDS)])
        abl_old = np.array([prior[i][f"ablation__{c}"] for i in range(N_SEEDS)])
        mis_old = np.array([prior[i][f"mismatch__{c}"] for i in range(N_SEEDS)])
        abl_new = np.array([r[f"ablation_rs__{c}"] for r in results])
        mis_new = np.array([r[f"mismatch_rs__{c}"] for r in results])
        _t, p_new = stats.ttest_rel(mis_new, pgd)
        print(f"{c:15s} {(abl_old-pgd).mean():+13.4f} {(abl_new-pgd).mean():+13.4f} "
              f"{(mis_old-pgd).mean():+9.4f} {(mis_new-pgd).mean():+9.4f} {p_new:11.5f}")

    print("\n=== Pooled over all conditions ===")
    def pool(key, src):
        return np.concatenate([[x[f"{key}__{c}"] for x in src] for c in EVAL_CONDITIONS])
    pgd_all = pool("pgd", prior)
    print(f"  scaffolding (ablation-PGD)  OLD: {(pool('ablation', prior)-pgd_all).mean():+.4f}   "
          f"NEW: {(pool('ablation_rs', results)-pgd_all).mean():+.4f}")
    print(f"  mechanism   (mism-ablation) OLD: {(pool('mismatch', prior)-pool('ablation', prior)).mean():+.4f}   "
          f"NEW: {(pool('mismatch_rs', results)-pool('ablation_rs', results)).mean():+.4f}")
    print(f"  NET         (mismatch-PGD)  OLD: {(pool('mismatch', prior)-pgd_all).mean():+.4f}   "
          f"NEW: {(pool('mismatch_rs', results)-pgd_all).mean():+.4f}")

    print("\nSaved per-seed results to hybridsn_random_start_results.json")


if __name__ == "__main__":
    main()
