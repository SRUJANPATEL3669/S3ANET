"""ASR sweep on SpectralFormer's actual architecture — patch-wise/CAF, not
the plain pixel-wise ViT used everywhere else in this project so far.

Every prior SpectralFormer result (`run_asr_sweep*.py`) targets the
pixel-wise ViT variant (`SpectralFormerRTAAAttack(band_patch=None)`) — a
simplification, not the group-wise-spectral-embedding + cross-layer-
adaptive-fusion (CAF) architecture the SpectralFormer paper actually
proposes. This script runs the same replicated protocol (RTAA mismatch-aware
vs. its own ablation vs. plain PGD, epsilon=0.01, 8 seeds, 4 evaluation
atmospheres, Bonferroni correction) against the real patch-wise/CAF model
(`variant="patchwise"`, band_patch=3, 7x7 spatial neighborhoods), to check
whether the pixel-wise result is representative of the architecture the
paper is actually making claims about, or an artifact of the simplified
variant.

Structural difference from the pixel-wise protocol: clean_spectra here is a
(B, 7, 7, n_bands) spatial neighborhood, not a (B, n_bands) single spectrum,
built via reflect-padding + neighborhood extraction from the same Indian
Pines cube (matching `gain_neighborhood_band`'s expected input). PGD is run
through a thin wrapper (`_WrappedClassifier`) that applies
`gain_neighborhood_band` internally, keeping PGD's output in the same raw
spatial-patch space RTAA attacks (needed for the atmospheric resimulation
step), mirroring the pattern used for HybridSN's PCA wrapper.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy import stats
from scipy.io import loadmat
from torch import Tensor, nn

from rtaa.attacks.baselines import pgd_attack
from rtaa.attacks.spectralformer_attack import (
    PhysicalViabilityWeights,
    SpectralFormerRTAAAttack,
    per_band_normalize,
)
from rtaa.models.spectralformer import gain_neighborhood_band, load_spectralformer_vit
from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state
from rtaa.rtm.surrogate import RTMSurrogate

SPECTRALFORMER_DATA = "/home/jayant/projects/SpectralFormer/data/IndianPine.mat"
RTM_CHECKPOINT = "checkpoints/rtm_surrogate_200bands.pt"
PATCH_SIZE = 7
BAND_PATCH = 3  # Indian Pines patch-wise near_band (Pavia uses 7, not relevant here)

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
    the group-wise spectral embedding SpectralFormer's CAF variant expects."""

    def __init__(self, classifier: nn.Module, band_patch: int):
        super().__init__()
        self.classifier = classifier
        self.band_patch = band_patch

    def forward(self, raw_patch: Tensor) -> Tensor:
        return self.classifier(gain_neighborhood_band(raw_patch, self.band_patch))


def extract_raw_patches(cube_norm: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch_size: int) -> np.ndarray:
    margin = patch_size // 2
    padded = np.pad(cube_norm, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
    n_bands = cube_norm.shape[-1]
    out = np.empty((len(rows), patch_size, patch_size, n_bands), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        rp, cp = r + margin, c + margin
        out[i] = padded[rp - margin : rp + margin + 1, cp - margin : cp + margin + 1, :]
    return out


def load_data(device: torch.device, n_samples: int, seed: int):
    d = loadmat(SPECTRALFORMER_DATA)
    input_cube = d["input"].astype(np.float64)
    TE = d["TE"]
    n_bands = input_cube.shape[2]

    cube_norm = per_band_normalize(input_cube.reshape(-1, n_bands)).reshape(input_cube.shape).astype(np.float32)
    rows_all, cols_all = np.nonzero(TE != 0)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(rows_all), min(n_samples, len(rows_all)), replace=False)
    rows, cols = rows_all[sel], cols_all[sel]
    labels_np = TE[rows, cols] - 1

    patches_np = extract_raw_patches(cube_norm, rows, cols, PATCH_SIZE)
    return (
        torch.from_numpy(patches_np).to(device),
        torch.from_numpy(labels_np).long().to(device),
        n_bands,
    )


def simulate_and_evaluate(
    adv_patches: Tensor, labels: Tensor, eval_atm_state_true: Tensor,
    surrogate: RTMSurrogate, solar: Tensor, wrapped_classifier: nn.Module,
) -> float:
    atm_state_assumed = perturb_atm_state(eval_atm_state_true, AtmosphericMismatchConfig())
    t_atm_true, l_path_true = surrogate(eval_atm_state_true)
    t_atm_assumed, l_path_assumed = surrogate(atm_state_assumed)
    t_atm_true_b = t_atm_true[:, None, None, :]
    l_path_true_b = l_path_true[:, None, None, :]
    t_atm_assumed_b = t_atm_assumed[:, None, None, :]
    l_path_assumed_b = l_path_assumed[:, None, None, :]

    l_sensor = sensor_radiance(adv_patches, t_atm_true_b, solar, l_path_true_b)
    r_rec = invert_to_reflectance(l_sensor, t_atm_assumed_b, solar, l_path_assumed_b)
    r_rec = torch.clamp(r_rec, 0.0, 1.0)

    with torch.no_grad():
        logits = wrapped_classifier(r_rec)
        acc = (logits.argmax(1) == labels).float().mean().item()
    return acc


def run_one_seed(seed: int, device: torch.device, surrogate: RTMSurrogate, classifier: nn.Module, wrapped_classifier: nn.Module) -> dict:
    torch.manual_seed(seed)
    clean_patches, labels, n_bands = load_data(device, N_SAMPLES, seed=seed)
    solar = (torch.rand(n_bands, generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.75).to(device)
    gen_atm_state = torch.tensor([GENERATION_ATM] * clean_patches.shape[0], device=device)

    rtaa_mismatch = SpectralFormerRTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), band_patch=BAND_PATCH,
        mismatch_config=AtmosphericMismatchConfig(),
    )
    adv_mismatch, _ = rtaa_mismatch.generate(classifier=classifier, clean_spectra=clean_patches, labels=labels, atm_state=gen_atm_state)

    rtaa_ablation = SpectralFormerRTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), band_patch=BAND_PATCH,
        mismatch_config=AtmosphericMismatchConfig.none(),
    )
    adv_ablation, _ = rtaa_ablation.generate(classifier=classifier, clean_spectra=clean_patches, labels=labels, atm_state=gen_atm_state)

    adv_pgd = pgd_attack(wrapped_classifier, clean_patches, labels, epsilon=EPSILON, step_size=EPSILON / 5, n_steps=N_STEPS)

    methods = {"mismatch": adv_mismatch, "ablation": adv_ablation, "pgd": adv_pgd}
    seed_result = {"seed": seed}
    for method_name, adv_patches in methods.items():
        for cond_name, cond_values in EVAL_CONDITIONS.items():
            eval_atm_state = torch.tensor([cond_values] * clean_patches.shape[0], device=device)
            adv_acc = simulate_and_evaluate(adv_patches, labels, eval_atm_state, surrogate, solar, wrapped_classifier)
            seed_result[f"{method_name}__{cond_name}"] = 1.0 - adv_acc
    return seed_result


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier = load_spectralformer_vit("Indian", variant="patchwise", device=device)
    classifier.eval()
    wrapped_classifier = _WrappedClassifier(classifier, BAND_PATCH).to(device)
    wrapped_classifier.eval()

    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=200).to(device)
    surrogate.eval()

    all_results = []
    for seed in range(N_SEEDS):
        print(f"=== seed {seed}/{N_SEEDS-1} ===")
        result = run_one_seed(seed, device, surrogate, classifier, wrapped_classifier)
        all_results.append(result)
        for cond_name in EVAL_CONDITIONS:
            m, a, p = result[f"mismatch__{cond_name}"], result[f"ablation__{cond_name}"], result[f"pgd__{cond_name}"]
            print(f"  {cond_name:15s} ASR: mismatch={m:.4f}  ablation={a:.4f}  pgd={p:.4f}  (mismatch-pgd={m-p:+.4f})")

    with open("asr_sweep_spectralformer_patchwise_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    n_conditions = len(EVAL_CONDITIONS)
    bonferroni_alpha = 0.05 / n_conditions
    print(f"\n=== Summary across {N_SEEDS} seeds (epsilon={EPSILON}), Bonferroni alpha={bonferroni_alpha:.4f} ===")
    for label_a, label_b in [("mismatch", "ablation"), ("mismatch", "pgd")]:
        print(f"\n--- RTAA (mismatch-aware) vs. {label_b} ---")
        for cond_name in EVAL_CONDITIONS:
            a_vals = np.array([r[f"{label_a}__{cond_name}"] for r in all_results])
            b_vals = np.array([r[f"{label_b}__{cond_name}"] for r in all_results])
            diff = a_vals - b_vals
            _t, p_value = stats.ttest_rel(a_vals, b_vals)
            sig = "***" if p_value < bonferroni_alpha else ("*" if p_value < 0.05 else "")
            print(f"{cond_name:15s} {label_a}={a_vals.mean():.4f}  {label_b}={b_vals.mean():.4f}  diff={diff.mean():+.4f}  p={p_value:.5f} {sig}")

    print("\nSaved per-seed results to asr_sweep_spectralformer_patchwise_results.json")


if __name__ == "__main__":
    main()
