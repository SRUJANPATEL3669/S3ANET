"""Gradient-level diagnostic for the HybridSN scaffolding cost -- the
"natural next step" flagged in PUBLICATION_ROADMAP.md/paper Limitations
after five optimizer-level hypotheses (ceiling effect, physical-viability
loss, momentum/normalization, random start, feasible-set mismatch) were
each tested and each refuted.

Dimensionality alone does not explain the pattern: SACNet has the largest
attack surface (~21M scalars, whole-scene) of any target and the best
scaffolding term (+0.022); HybridSN has a far smaller attack surface
(125,000 scalars, a 25x25x200 patch) and the worst (-0.109). What HybridSN
uniquely has, that no other target does, is a differentiable PCA/whitening
projection between delta and the classifier -- RTAAAttack perturbs raw
reflectance, which is then projected through `DifferentiablePCA` before
reaching HybridSN; every other target's classifier consumes the perturbed
input closer to directly.

This script computes the raw (pre-normalization) gradient reaching delta,
at the identical starting point (delta=0, same clean patches, same labels,
same classifier) for RTAA's full pipeline (RTM forward -> RTM inverse ->
PCA -> classifier) vs. PGD's path (PCA -> classifier, no RTM chain) on
HybridSN, and reports:
  - gradient L2 norm (is RTAA's gradient systematically smaller/larger?)
  - per-band gradient concentration (Gini-style: fraction of gradient mass
    in the top-10% of components, by |grad| magnitude) -- does the PCA
    chain make RTAA's gradient sparser/more concentrated than PGD's?
  - cosine similarity between the two gradients (do they even point in
    similar directions before either optimizer's sign/momentum machinery
    touches them?)
"""

from __future__ import annotations

import numpy as np
import torch
from run_asr_sweep_hybridsn import (
    CHECKPOINT,
    GENERATION_ATM,
    N_SAMPLES,
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
    stratified_train_test_split,
)

from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state

N_SEEDS = 8


def concentration(grad: torch.Tensor) -> float:
    flat = grad.abs().flatten()
    total = flat.sum().item()
    if total == 0:
        return float("nan")
    k = max(1, int(0.10 * flat.numel()))
    top_k = torch.topk(flat, k).values.sum().item()
    return top_k / total


def one_seed(seed, test_rows, test_cols, test_labels, cube_norm, device, surrogate, pca_projector, hybridsn, wrapped):
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

    # --- PGD's gradient: delta=0, straight through PCA -> classifier ---
    delta_pgd = torch.zeros_like(clean_patches, requires_grad=True)
    logits_pgd = wrapped(clean_patches + delta_pgd)
    loss_pgd = torch.nn.functional.cross_entropy(logits_pgd, labels)
    grad_pgd = torch.autograd.grad(loss_pgd, delta_pgd)[0]

    # --- RTAA's gradient: delta=0, through RTM forward -> RTM inverse -> PCA -> classifier ---
    delta_rtaa = torch.zeros_like(clean_patches, requires_grad=True)
    mismatch_config = AtmosphericMismatchConfig()
    atm_state_assumed = perturb_atm_state(gen_atm_state, mismatch_config)
    t_atm_true, l_path_true = surrogate(gen_atm_state)
    t_atm_assumed, l_path_assumed = surrogate(atm_state_assumed)

    def bshape(x):
        extra = clean_patches.ndim - x.ndim
        return x.reshape(x.shape[:1] + (1,) * extra + x.shape[1:])

    r_adv = torch.clamp(clean_patches + delta_rtaa, 0.0, 1.0)
    l_sensor = sensor_radiance(r_adv, bshape(t_atm_true), solar, bshape(l_path_true))
    r_rec = invert_to_reflectance(l_sensor, bshape(t_atm_assumed), solar, bshape(l_path_assumed))
    pca_patch = pca_projector(r_rec).permute(0, 3, 1, 2).unsqueeze(1)
    logits_rtaa = hybridsn(pca_patch)
    loss_rtaa = torch.nn.functional.cross_entropy(logits_rtaa, labels)
    grad_rtaa = torch.autograd.grad(loss_rtaa, delta_rtaa)[0]

    cos_sim = torch.nn.functional.cosine_similarity(grad_pgd.flatten(), grad_rtaa.flatten(), dim=0).item()

    return {
        "seed": seed,
        "grad_norm_pgd": grad_pgd.norm().item(),
        "grad_norm_rtaa": grad_rtaa.norm().item(),
        "grad_mean_abs_pgd": grad_pgd.abs().mean().item(),
        "grad_mean_abs_rtaa": grad_rtaa.abs().mean().item(),
        "concentration_pgd": concentration(grad_pgd),
        "concentration_rtaa": concentration(grad_rtaa),
        "cosine_similarity": cos_sim,
    }


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
    import json

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

    results = []
    for seed in range(N_SEEDS):
        r = one_seed(seed, test_rows, test_cols, test_labels, cube_norm, device, surrogate, pca_projector, hybridsn, wrapped)
        results.append(r)
        print(f"seed {seed}: ||grad||_PGD={r['grad_norm_pgd']:.6f}  ||grad||_RTAA={r['grad_norm_rtaa']:.6f}  "
              f"ratio={r['grad_norm_rtaa']/r['grad_norm_pgd']:.4f}  cos_sim={r['cosine_similarity']:.4f}  "
              f"conc_PGD={r['concentration_pgd']:.4f}  conc_RTAA={r['concentration_rtaa']:.4f}")

    print(f"\n=== Summary across {N_SEEDS} seeds ===")
    for key in ["grad_norm_pgd", "grad_norm_rtaa", "concentration_pgd", "concentration_rtaa", "cosine_similarity"]:
        vals = np.array([r[key] for r in results])
        print(f"  {key:20s} mean={vals.mean():.5f}  std={vals.std(ddof=1):.5f}")
    ratios = np.array([r["grad_norm_rtaa"] / r["grad_norm_pgd"] for r in results])
    print(f"\n  grad norm ratio (RTAA/PGD): mean={ratios.mean():.4f}  std={ratios.std(ddof=1):.4f}")

    with open("hybridsn_gradient_diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to hybridsn_gradient_diagnostic_results.json")


if __name__ == "__main__":
    main()
