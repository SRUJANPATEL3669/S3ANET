"""Sixth candidate fix for the HybridSN scaffolding cost, this time not a
refutation attempt but a positive proposal, directly motivated by the
gradient diagnostic (PUBLICATION_ROADMAP.md): HybridSN's gradient
concentration is 0.999 (a handful of dimensions carry ~all the gradient
mass), so RTAAAttack's mean-based normalization (`grad / grad.abs().mean()`)
is dominated by those outliers every step, and momentum accumulates
whatever direction they point in. Tests `grad_normalization="median"`
(rtaa_attack.RTAAAttack), a robust alternative unaffected by a small number
of extreme-magnitude dimensions, against the original "mean" config and
against PGD, on HybridSN, same 8-seed protocol as the other diagnostics.
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

    attack_median = RTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig(),
        grad_normalization="median",
    )
    adv_median, _ = attack_median.generate(
        classifier=hybridsn, pca_projector=pca_projector, clean_spectra=clean_patches,
        clean_patch_for_shape=clean_patches, labels=labels, atm_state=gen_atm_state,
    )

    seed_result = {"seed": seed}
    for cond_name, cond_values in EVAL_CONDITIONS.items():
        eval_atm = torch.tensor([cond_values] * clean_patches.shape[0], device=device)
        acc = simulate_and_evaluate_patches(adv_median, labels, eval_atm, surrogate, solar, wrapped_classifier)
        seed_result[f"mismatch_median__{cond_name}"] = 1.0 - acc
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
            print(f"  {c:15s} median={r[f'mismatch_median__{c}']:.4f}  "
                  f"(prior mean-norm mismatch={prior[seed][f'mismatch__{c}']:.4f}, pgd={prior[seed][f'pgd__{c}']:.4f})")

    with open("hybridsn_median_norm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    alpha = 0.05 / len(EVAL_CONDITIONS)
    print(f"\n=== HybridSN: median-normalized RTAA vs. PGD vs. original mean-normalized RTAA (Bonferroni alpha={alpha:.4f}) ===")
    for c in EVAL_CONDITIONS:
        pgd = np.array([prior[i][f"pgd__{c}"] for i in range(N_SEEDS)])
        mean_norm = np.array([prior[i][f"mismatch__{c}"] for i in range(N_SEEDS)])
        median_norm = np.array([r[f"mismatch_median__{c}"] for r in results])
        _t1, p_vs_pgd = stats.ttest_rel(median_norm, pgd)
        _t2, p_vs_mean = stats.ttest_rel(median_norm, mean_norm)
        print(f"{c:15s} mean-norm={mean_norm.mean():.4f}  median-norm={median_norm.mean():.4f}  pgd={pgd.mean():.4f}   "
              f"(median-pgd diff={median_norm.mean()-pgd.mean():+.4f}, p={p_vs_pgd:.5f})   "
              f"(median-mean diff={median_norm.mean()-mean_norm.mean():+.4f}, p={p_vs_mean:.5f})")

    print("\n=== Pooled ===")
    def pool(key, src):
        return np.concatenate([[x[f"{key}__{c}"] for x in src] for c in EVAL_CONDITIONS])
    pgd_all = pool("pgd", prior)
    mean_all = pool("mismatch", prior)
    median_all = pool("mismatch_median", results)
    print(f"  scaffolding-like term (median-norm RTAA - PGD): {(median_all-pgd_all).mean():+.4f}   "
          f"(original mean-norm RTAA - PGD): {(mean_all-pgd_all).mean():+.4f}")

    print("\nSaved per-seed results to hybridsn_median_norm_results.json")


if __name__ == "__main__":
    main()
