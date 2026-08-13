"""Cross-representation transferability: RTAA attacks crafted against
SpectralFormer, evaluated on HybridSN -- neither previously tested. The
existing transferability matrix (`run_transferability_matrix.py`) only
covers SACNet<->S3ANet, which share an identical whole-scene data pipeline;
SpectralFormer and HybridSN are architecturally harder to bridge (single
spectrum vs. spatial patch, different per-band vs. global normalization),
but they ARE both trained on the same physical Indian Pines scene, so a
transfer test is possible with some bridging, unlike the cross-dataset
(PaviaU vs. Indian Pines) cases that were genuinely out of scope.

Two real obstacles found and resolved while building this:
(1) SpectralFormer's own data file and this project's SAFER-mirror Indian
    Pines file are 96.7% pixel-identical (same scene, tiny distributor
    differences) but use COMPLETELY DIFFERENT class-index numbering for the
    ground truth (0% raw agreement) -- confirmed via a 100%-agreement
    per-class majority-vote mapping (SAFER_TO_SF below). Silently mixing the
    two would have quietly evaluated against the wrong class each time.
(2) SpectralFormer normalizes per-band (min-max per wavelength across the
    whole cube); HybridSN normalizes by a single global max. An adversarial
    spectrum generated in one convention isn't valid input to the other
    without converting through raw units first.

Both resolved by working from ONE canonical cube (this project's SAFER-
mirror source, since that's what HybridSN's checkpoint/PCA were fit on),
applying each classifier's own normalization formula to it, and explicitly
converting the SpectralFormer-generated adversarial perturbation back to
raw units before re-normalizing into HybridSN's convention.

Two transfer variants, per the user's request:
  (A) SPECTRAL-ONLY: attack generated against SpectralFormer pixel-wise
      (single spectrum, no spatial context) -- transferred by substituting
      the perturbed center-pixel spectrum into an otherwise-clean HybridSN
      patch. Carries no spatial structure, since pixel-wise SpectralFormer
      never touches neighboring pixels.
  (B) SPATIAL: attack generated against SpectralFormer's actual patch-wise/
      CAF architecture (7x7 neighborhood) -- transferred by embedding the
      full perturbed 7x7 region into the center of HybridSN's 25x25 patch.

Includes a RANDOM-PERTURBATION control (same epsilon, uniform noise, same
embedding procedure) at each transfer point, since a transfer number in
isolation doesn't establish anything -- it needs to be compared against
what an unstructured perturbation of the same magnitude achieves.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from run_asr_sweep_hybridsn import (
    CHECKPOINT,
    GENERATION_ATM,
    N_STEPS,
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
    stratified_train_test_split,
)
from run_asr_sweep_hybridsn import (
    PATCH_SIZE as HYBRIDSN_PATCH_SIZE,
)
from scipy import stats

from rtaa.attacks.spectralformer_attack import (
    PhysicalViabilityWeights,
    SpectralFormerRTAAAttack,
    per_band_normalize,
)
from rtaa.models.spectralformer import load_spectralformer_vit
from rtaa.rtm.mismatch import AtmosphericMismatchConfig

SAFER_TO_SF = {1: 14, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 15, 8: 6, 9: 16, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 13}

SF_PATCH_SIZE = 7
SF_BAND_PATCH = 3
EPSILON = 0.01
N_TRANSFER_SAMPLES = 300
N_SEEDS = 8


def build_transfer_pieces(device):
    cube, labels_map = load_hsi_cube("IndianPines")
    n_bands = cube.shape[-1]

    cube_norm_sf = per_band_normalize(cube.reshape(-1, n_bands)).reshape(cube.shape).astype(np.float32)
    cube_norm_hybridsn = normalize_reflectance(cube).astype(np.float32)

    flat = cube.reshape(-1, n_bands)
    lo, hi = flat.min(axis=0), flat.max(axis=0)  # SF's per-band normalization params
    global_max = cube.max()  # HybridSN's global normalization param

    return cube, labels_map, cube_norm_sf, cube_norm_hybridsn, lo.astype(np.float32), hi.astype(np.float32), np.float32(global_max)


def sf_normalized_to_hybridsn_normalized(x_sf: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, global_max: float) -> torch.Tensor:
    """Convert a tensor in SpectralFormer's per-band-normalized domain back
    to raw units, then into HybridSN's global-max-normalized domain."""
    raw = x_sf * (hi - lo) + lo
    return raw / global_max


def run_one_seed(seed, cube, labels_map, cube_norm_sf, cube_norm_hybridsn, lo, hi, global_max,
                  test_rows_hyb, test_cols_hyb, test_labels_hyb,
                  device, surrogate, sf_pixel_clf, sf_patch_clf, wrapped_hybridsn):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(test_rows_hyb), min(N_TRANSFER_SAMPLES, len(test_rows_hyb)), replace=False)
    rows, cols = test_rows_hyb[sel], test_cols_hyb[sel]
    safer_labels = test_labels_hyb[sel]
    sf_labels_np = np.array([SAFER_TO_SF[int(c)] for c in safer_labels]) - 1
    hyb_labels_np = safer_labels - 1

    lo_t = torch.from_numpy(lo).to(device)
    hi_t = torch.from_numpy(hi).to(device)

    # HybridSN's clean patches (25x25, its own normalization) -- the base every transfer embeds into
    clean_hyb_patches = torch.from_numpy(
        extract_raw_patches(cube_norm_hybridsn, rows, cols, HYBRIDSN_PATCH_SIZE)
    ).to(device)
    hyb_labels = torch.from_numpy(hyb_labels_np).long().to(device)

    solar = (torch.rand(cube.shape[-1], generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.75).to(device)
    n_batch = clean_hyb_patches.shape[0]
    gen_atm_state = torch.tensor([GENERATION_ATM] * n_batch, device=device)

    center_hyb = HYBRIDSN_PATCH_SIZE // 2
    result = {"seed": seed}

    with torch.no_grad():
        clean_acc = (wrapped_hybridsn(clean_hyb_patches).argmax(1) == hyb_labels).float().mean().item()
    result["clean_acc"] = clean_acc

    # ---------------- (A) spectral-only: pixel-wise SF -> HybridSN center pixel ----------------
    sf_labels = torch.from_numpy(sf_labels_np).long().to(device)
    clean_sf_spectra = torch.from_numpy(cube_norm_sf[rows, cols, :]).to(device)

    attack_pixel = SpectralFormerRTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig(),
    )
    adv_sf_spectra, _ = attack_pixel.generate(
        classifier=sf_pixel_clf, clean_spectra=clean_sf_spectra, labels=sf_labels, atm_state=gen_atm_state,
    )
    adv_center_hyb = sf_normalized_to_hybridsn_normalized(adv_sf_spectra, lo_t, hi_t, float(global_max))

    transferred_a = clean_hyb_patches.clone()
    transferred_a[:, center_hyb, center_hyb, :] = torch.clamp(adv_center_hyb, 0.0, 1.0)
    with torch.no_grad():
        acc_a = (wrapped_hybridsn(transferred_a).argmax(1) == hyb_labels).float().mean().item()
    result["asr_spectral_transfer"] = 1.0 - acc_a

    # random-noise control, same epsilon, same embedding point
    noise = torch.empty_like(clean_sf_spectra).uniform_(-EPSILON, EPSILON)
    adv_random_sf = torch.clamp(clean_sf_spectra + noise, 0.0, 1.0)
    adv_random_hyb = sf_normalized_to_hybridsn_normalized(adv_random_sf, lo_t, hi_t, float(global_max))
    transferred_a_rand = clean_hyb_patches.clone()
    transferred_a_rand[:, center_hyb, center_hyb, :] = torch.clamp(adv_random_hyb, 0.0, 1.0)
    with torch.no_grad():
        acc_a_rand = (wrapped_hybridsn(transferred_a_rand).argmax(1) == hyb_labels).float().mean().item()
    result["asr_spectral_random_control"] = 1.0 - acc_a_rand

    # ---------------- (B) spatial: patch-wise/CAF SF -> HybridSN center 7x7 ----------------
    clean_sf_patches = torch.from_numpy(
        extract_raw_patches(cube_norm_sf, rows, cols, SF_PATCH_SIZE)
    ).to(device)

    attack_patch = SpectralFormerRTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), band_patch=SF_BAND_PATCH,
        mismatch_config=AtmosphericMismatchConfig(),
    )
    adv_sf_patch, _ = attack_patch.generate(
        classifier=sf_patch_clf, clean_spectra=clean_sf_patches, labels=sf_labels, atm_state=gen_atm_state,
    )
    adv_patch_hyb = sf_normalized_to_hybridsn_normalized(adv_sf_patch, lo_t, hi_t, float(global_max))
    adv_patch_hyb = torch.clamp(adv_patch_hyb, 0.0, 1.0)

    half = SF_PATCH_SIZE // 2
    lo_idx, hi_idx = center_hyb - half, center_hyb + half + 1
    transferred_b = clean_hyb_patches.clone()
    transferred_b[:, lo_idx:hi_idx, lo_idx:hi_idx, :] = adv_patch_hyb
    with torch.no_grad():
        acc_b = (wrapped_hybridsn(transferred_b).argmax(1) == hyb_labels).float().mean().item()
    result["asr_spatial_transfer"] = 1.0 - acc_b

    noise_patch = torch.empty_like(clean_sf_patches).uniform_(-EPSILON, EPSILON)
    adv_random_patch_sf = torch.clamp(clean_sf_patches + noise_patch, 0.0, 1.0)
    adv_random_patch_hyb = torch.clamp(sf_normalized_to_hybridsn_normalized(adv_random_patch_sf, lo_t, hi_t, float(global_max)), 0.0, 1.0)
    transferred_b_rand = clean_hyb_patches.clone()
    transferred_b_rand[:, lo_idx:hi_idx, lo_idx:hi_idx, :] = adv_random_patch_hyb
    with torch.no_grad():
        acc_b_rand = (wrapped_hybridsn(transferred_b_rand).argmax(1) == hyb_labels).float().mean().item()
    result["asr_spatial_random_control"] = 1.0 - acc_b_rand

    return result


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cube, labels_map, cube_norm_sf, cube_norm_hybridsn, lo, hi, global_max = build_transfer_pieces(device)

    rows_all, cols_all = np.nonzero(labels_map != 0)
    entry_labels = labels_map[rows_all, cols_all].tolist()
    _tr, test_idx = stratified_train_test_split(entry_labels, TRAIN_FRACTION, SPLIT_SEED)
    test_rows, test_cols = rows_all[test_idx], cols_all[test_idx]
    test_labels = labels_map[test_rows, test_cols]

    pca_projector = build_pca_projector(cube_norm_hybridsn, PCA_COMPONENTS, device)
    with open(CHECKPOINT.replace(".pt", ".json")) as f:
        meta = json.load(f)
    hybridsn = HybridSN(n_bands=meta["n_bands"], n_classes=meta["n_classes"],
                        patch_size=meta["patch_size"], pca_components=meta["pca_components"]).to(device)
    hybridsn.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    hybridsn.eval()
    wrapped_hybridsn = _WrappedClassifier(pca_projector, hybridsn).to(device)
    wrapped_hybridsn.eval()

    sf_pixel_clf = load_spectralformer_vit("Indian", variant="pixelwise", device=device)
    sf_pixel_clf.eval()
    sf_patch_clf = load_spectralformer_vit("Indian", variant="patchwise", device=device)
    sf_patch_clf.eval()

    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=200).to(device)
    surrogate.eval()

    results = []
    for seed in range(N_SEEDS):
        print(f"=== seed {seed}/{N_SEEDS-1} ===")
        r = run_one_seed(seed, cube, labels_map, cube_norm_sf, cube_norm_hybridsn, lo, hi, global_max,
                          test_rows, test_cols, test_labels, device, surrogate, sf_pixel_clf, sf_patch_clf, wrapped_hybridsn)
        results.append(r)
        print(f"  clean_acc={r['clean_acc']:.4f}")
        print(f"  spectral transfer ASR={r['asr_spectral_transfer']:.4f}  (random control={r['asr_spectral_random_control']:.4f})")
        print(f"  spatial  transfer ASR={r['asr_spatial_transfer']:.4f}  (random control={r['asr_spatial_random_control']:.4f})")

    with open("transfer_spectralformer_to_hybridsn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Summary across {N_SEEDS} seeds ===")
    for key_pair, label in [
        (("asr_spectral_transfer", "asr_spectral_random_control"), "(A) SPECTRAL-ONLY: pixel-wise SF -> HybridSN center pixel"),
        (("asr_spatial_transfer", "asr_spatial_random_control"), "(B) SPATIAL: patch-wise/CAF SF -> HybridSN center 7x7"),
    ]:
        transfer_key, control_key = key_pair
        transfer_vals = np.array([r[transfer_key] for r in results])
        control_vals = np.array([r[control_key] for r in results])
        diff = transfer_vals - control_vals
        _t, p = stats.ttest_rel(transfer_vals, control_vals)
        print(f"\n{label}")
        print(f"  RTAA-transfer ASR:    {transfer_vals.mean():.4f} +- {transfer_vals.std(ddof=1):.4f}")
        print(f"  random-noise control: {control_vals.mean():.4f} +- {control_vals.std(ddof=1):.4f}")
        print(f"  diff (transfer - random): {diff.mean():+.4f}  p={p:.5f}")

    clean_vals = np.array([r["clean_acc"] for r in results])
    print(f"\nClean accuracy (HybridSN, transferred clean patches): {clean_vals.mean():.4f} +- {clean_vals.std(ddof=1):.4f}")
    print("\nSaved per-seed results to transfer_spectralformer_to_hybridsn_results.json")


if __name__ == "__main__":
    main()
