"""ASR sweep on a second architecture — HybridSN / Indian Pines.

Every ASR result so far (PUBLICATION_ROADMAP.md §1) is on SpectralFormer
pixel-wise / Indian Pines only. Before claiming this generalizes, the same
8-seed replicated protocol (RTAA mismatch-aware vs. RTAA ablation vs. plain
PGD, epsilon=0.01, same 4 evaluation atmospheres) needs to reproduce on a
structurally different classifier. HybridSN is the natural second target:
already trained (`checkpoints/hybridsn_indianpines.pt`, 99.9% test acc), and
a 3D-2D CNN over PCA'd spatial patches is about as architecturally different
from a pixel-wise transformer as it gets while still being cheap to run.

Structural differences from the SpectralFormer protocol, handled here:
- HybridSN consumes (patch, patch, pca_components) neighborhoods, not single
  per-band-normalized spectra. The attack (`RTAAAttack`, not
  `SpectralFormerRTAAAttack`) perturbs raw [0,1] reflectance *patches*, then
  projects through a `DifferentiablePCA` fitted to match the classifier's
  training-time PCA before feeding the classifier — this is what
  `rtaa.attacks.rtaa_attack.RTAAAttack` was built for.
- PGD (baseline) is run directly on the same raw-reflectance patch
  representation via a thin wrapper module (`_WrappedClassifier`) that
  applies the PCA + classifier internally — this keeps PGD's output in the
  same space as RTAA's, which the atmospheric resimulation step requires.
- Test-set patches are sampled from the *actual held-out test split* used to
  train the checkpoint (`stratified_train_test_split`, train_fraction=0.7,
  seed=0 — reconstructed here, not re-derived from a different split), so
  this isn't evaluating on data the classifier saw during training.
- PCA is refit here rather than loaded from the checkpoint (none was saved).
  This is safe: `HSIPatchDataset.apply_pca` fits deterministically from the
  same cube, and PCA-with-whitening is invariant to a global positive
  rescale (see `hsi_dataset.normalize_reflectance` docstring) — training used
  the raw-DN cube, this uses the [0,1]-normalized cube, and whitened PCA
  output is identical either way.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from torch import Tensor, nn

from rtaa.attacks.baselines import pgd_attack
from rtaa.attacks.rtaa_attack import (
    DifferentiablePCA,
    PhysicalViabilityWeights,
    RTAAAttack,
)
from rtaa.data.hsi_dataset import load_hsi_cube, normalize_reflectance
from rtaa.models.hybridsn import HybridSN
from rtaa.models.train_classifier import stratified_train_test_split
from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state
from rtaa.rtm.surrogate import RTMSurrogate

CHECKPOINT = "checkpoints/hybridsn_indianpines.pt"
RTM_CHECKPOINT = "checkpoints/rtm_surrogate_200bands.pt"
PATCH_SIZE = 25
PCA_COMPONENTS = 30
TRAIN_FRACTION = 0.7
SPLIT_SEED = 0  # matches train_classifier.py's default at checkpoint-training time

GENERATION_ATM = [0.05, 0.8, 15.0]
EVAL_CONDITIONS = {
    "clear": [0.05, 0.8, 15.0],
    "moderate_haze": [0.2, 1.5, 30.0],
    "heavy_haze": [0.4, 2.5, 45.0],
    "extreme": [0.5, 4.0, 60.0],
}
EPSILON = 0.01
N_SAMPLES = 300
N_STEPS = 25
N_SEEDS = 8


class _WrappedClassifier(nn.Module):
    """raw [0,1] reflectance patch (B, patch, patch, n_bands) -> logits, via
    the same PCA + reshape HybridSN expects. Lets the generic pixel-domain
    baselines (`rtaa.attacks.baselines`) operate in the same raw-reflectance
    space RTAA does, instead of their usual post-PCA convention."""

    def __init__(self, pca_projector: DifferentiablePCA, hybridsn: HybridSN):
        super().__init__()
        self.pca_projector = pca_projector
        self.hybridsn = hybridsn

    def forward(self, raw_patch: Tensor) -> Tensor:
        pca_patch = self.pca_projector(raw_patch).permute(0, 3, 1, 2).unsqueeze(1)
        return self.hybridsn(pca_patch)


def build_pca_projector(cube_norm: np.ndarray, n_components: int, device: torch.device) -> DifferentiablePCA:
    n_bands = cube_norm.shape[-1]
    flat = cube_norm.reshape(-1, n_bands)
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(flat)
    mean = torch.from_numpy(pca.mean_.astype(np.float32)).to(device)
    components = torch.from_numpy(pca.components_.astype(np.float32)).to(device)
    whiten_scale = torch.from_numpy(np.sqrt(pca.explained_variance_).astype(np.float32)).to(device)
    return DifferentiablePCA(mean, components, whiten_scale)


def extract_raw_patches(cube_norm: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch_size: int) -> np.ndarray:
    margin = patch_size // 2
    padded = np.pad(cube_norm, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
    n_bands = cube_norm.shape[-1]
    out = np.empty((len(rows), patch_size, patch_size, n_bands), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        rp, cp = r + margin, c + margin
        out[i] = padded[rp - margin : rp + margin + 1, cp - margin : cp + margin + 1, :]
    return out


def simulate_and_evaluate_patches(
    adv_patches: Tensor,
    labels: Tensor,
    eval_atm_state_true: Tensor,
    surrogate: RTMSurrogate,
    solar: Tensor,
    wrapped_classifier: nn.Module,
) -> float:
    atm_state_assumed = perturb_atm_state(eval_atm_state_true, AtmosphericMismatchConfig())
    t_atm_true, l_path_true = surrogate(eval_atm_state_true)
    t_atm_assumed, l_path_assumed = surrogate(atm_state_assumed)

    l_sensor = sensor_radiance(adv_patches, t_atm_true[:, None, None, :], solar, l_path_true[:, None, None, :])
    r_rec = invert_to_reflectance(l_sensor, t_atm_assumed[:, None, None, :], solar, l_path_assumed[:, None, None, :])
    r_rec = torch.clamp(r_rec, 0.0, 1.0)

    with torch.no_grad():
        logits = wrapped_classifier(r_rec)
        acc = (logits.argmax(1) == labels).float().mean().item()
    return acc


def run_one_seed(
    seed: int, test_rows: np.ndarray, test_cols: np.ndarray, test_labels: np.ndarray,
    cube_norm: np.ndarray, device: torch.device, surrogate: RTMSurrogate,
    pca_projector: DifferentiablePCA, hybridsn: HybridSN, wrapped_classifier: nn.Module,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(test_rows), min(N_SAMPLES, len(test_rows)), replace=False)
    rows, cols = test_rows[sel], test_cols[sel]
    labels_np = test_labels[sel] - 1  # 1-indexed -> 0-indexed

    raw_patches_np = extract_raw_patches(cube_norm, rows, cols, PATCH_SIZE)
    clean_patches = torch.from_numpy(raw_patches_np).to(device)
    labels = torch.from_numpy(labels_np).long().to(device)

    n_bands = cube_norm.shape[-1]
    solar = (torch.rand(n_bands, generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.75).to(device)
    gen_atm_state = torch.tensor([GENERATION_ATM] * clean_patches.shape[0], device=device)

    rtaa_mismatch = RTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig(),
    )
    adv_mismatch, _ = rtaa_mismatch.generate(
        classifier=hybridsn, pca_projector=pca_projector, clean_spectra=clean_patches,
        clean_patch_for_shape=clean_patches, labels=labels, atm_state=gen_atm_state,
    )

    rtaa_ablation = RTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig.none(),
    )
    adv_ablation, _ = rtaa_ablation.generate(
        classifier=hybridsn, pca_projector=pca_projector, clean_spectra=clean_patches,
        clean_patch_for_shape=clean_patches, labels=labels, atm_state=gen_atm_state,
    )

    adv_pgd = pgd_attack(
        wrapped_classifier, clean_patches, labels, epsilon=EPSILON, step_size=EPSILON / 5, n_steps=N_STEPS,
    )

    methods = {"mismatch": adv_mismatch, "ablation": adv_ablation, "pgd": adv_pgd}
    seed_result = {"seed": seed}
    for method_name, adv_patches in methods.items():
        for cond_name, cond_values in EVAL_CONDITIONS.items():
            eval_atm_state = torch.tensor([cond_values] * clean_patches.shape[0], device=device)
            adv_acc = simulate_and_evaluate_patches(adv_patches, labels, eval_atm_state, surrogate, solar, wrapped_classifier)
            seed_result[f"{method_name}__{cond_name}"] = 1.0 - adv_acc
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
    print(f"Reconstructed test split: {len(test_idx)} pixels (expect 3081 per checkpoint metadata)")

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
        result = run_one_seed(
            seed, test_rows, test_cols, test_labels, cube_norm, device,
            surrogate, pca_projector, hybridsn, wrapped_classifier,
        )
        all_results.append(result)
        for cond_name in EVAL_CONDITIONS:
            m, a, p = result[f"mismatch__{cond_name}"], result[f"ablation__{cond_name}"], result[f"pgd__{cond_name}"]
            print(f"  {cond_name:15s} ASR: mismatch={m:.4f}  ablation={a:.4f}  pgd={p:.4f}  (mismatch-pgd={m-p:+.4f})")

    with open("asr_sweep_hybridsn_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    n_conditions = len(EVAL_CONDITIONS)
    bonferroni_alpha = 0.05 / n_conditions
    print(f"\n=== Summary across {N_SEEDS} seeds (epsilon={EPSILON}), Bonferroni alpha={bonferroni_alpha:.4f} ===")
    for cond_name in EVAL_CONDITIONS:
        mismatch_vals = np.array([r[f"mismatch__{cond_name}"] for r in all_results])
        pgd_vals = np.array([r[f"pgd__{cond_name}"] for r in all_results])
        diff = mismatch_vals - pgd_vals

        _t_stat, p_value = stats.ttest_rel(mismatch_vals, pgd_vals)
        sig = "***" if p_value < bonferroni_alpha else ("*" if p_value < 0.05 else "")

        print(f"\n{cond_name}:")
        print(f"  RTAA (mismatch-aware): {mismatch_vals.mean():.4f} +- {mismatch_vals.std(ddof=1):.4f}")
        print(f"  PGD (baseline):        {pgd_vals.mean():.4f} +- {pgd_vals.std(ddof=1):.4f}")
        print(f"  mismatch - pgd:        {diff.mean():+.4f} +- {diff.std(ddof=1):.4f}  (paired t-test p={p_value:.5f}) {sig}")

    print("\nSaved per-seed results to asr_sweep_hybridsn_results.json")


if __name__ == "__main__":
    main()
