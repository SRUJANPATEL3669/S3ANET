"""Atmospheric Survival Rate (ASR) sweep — PUBLICATION_ROADMAP.md §1.

Never run before now. This is the experiment that actually tests whether
RTAA's atmospheric-mismatch-aware attack generation produces a *better*
attack, not just a *different* one (the earlier ablation spot-check showed
the mechanism changes results but hit a ceiling effect at eps=0.05 on both
configs).

Protocol: generate adversarial reflectance under one "generation" atmosphere
using three methods (RTAA with mismatch, RTAA ablation/no-mismatch, and a
plain pixel-domain PGD baseline with no physics at all). Treat the resulting
perturbed reflectance as the physical surface change in every case. Then, for
each of several "evaluation" atmospheres, resimulate how that physical
surface would actually be sensed and compensated under that condition, and
measure classifier accuracy. ASR = fraction of adversarial examples that
remain misclassified (1 - accuracy) under each evaluation condition.

Target: SpectralFormer pixel-wise ViT on Indian Pines. Chosen because its
input format IS raw per-band-normalized reflectance (no PCA), so the plain
PGD baseline's output is directly reusable as "physical reflectance" for the
resimulation step — an apples-to-apples comparison across all three methods
without needing a PCA round-trip for the baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.io import loadmat

from rtaa.attacks.baselines import pgd_attack
from rtaa.attacks.spectralformer_attack import (
    PhysicalViabilityWeights,
    SpectralFormerRTAAAttack,
    per_band_normalize,
)
from rtaa.models.spectralformer import load_spectralformer_vit
from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state
from rtaa.rtm.surrogate import RTMSurrogate

SPECTRALFORMER_DATA = "/home/jayant/projects/SpectralFormer/data/IndianPine.mat"
RTM_CHECKPOINT = "checkpoints/rtm_surrogate_200bands.pt"

# atm_state columns: [aerosol_optical_depth, water_vapor_cm, solar_zenith_deg]
GENERATION_ATM = [0.05, 0.8, 15.0]  # attacker crafts assuming clear conditions
EVAL_CONDITIONS = {
    "clear": [0.05, 0.8, 15.0],
    "moderate_haze": [0.2, 1.5, 30.0],
    "heavy_haze": [0.4, 2.5, 45.0],
    "extreme": [0.5, 4.0, 60.0],
}
EPSILON_SWEEP = [0.01, 0.03, 0.05, 0.1]
N_SAMPLES = 500
N_STEPS = 25


@dataclass
class ASRResult:
    method: str
    epsilon: float
    eval_condition: str
    clean_acc: float
    adv_acc: float
    asr: float  # 1 - adv_acc


def load_data(device: torch.device, n_samples: int, seed: int = 0):
    d = loadmat(SPECTRALFORMER_DATA)
    input_cube = d["input"].astype(np.float64)
    TE = d["TE"]
    n_bands = input_cube.shape[2]

    flat_norm = per_band_normalize(input_cube.reshape(-1, n_bands)).reshape(input_cube.shape)
    rows, cols = np.nonzero(TE != 0)
    sel = np.random.default_rng(seed).choice(len(rows), min(n_samples, len(rows)), replace=False)
    spectra = np.stack([flat_norm[rows[i], cols[i], :] for i in sel]).astype(np.float32)
    labels = np.array([TE[rows[i], cols[i]] - 1 for i in sel])

    return (
        torch.from_numpy(spectra).to(device),
        torch.from_numpy(labels).long().to(device),
        n_bands,
    )


def simulate_and_evaluate(
    adv_reflectance: torch.Tensor,
    labels: torch.Tensor,
    eval_atm_state_true: torch.Tensor,
    surrogate: RTMSurrogate,
    solar: torch.Tensor,
    classifier: torch.nn.Module,
) -> float:
    """Resimulates adv_reflectance (the physical surface) being sensed and
    compensated under eval_atm_state_true, with realistic compensation error
    (AtmosphericMismatchConfig default). Returns accuracy under that condition."""
    atm_state_assumed = perturb_atm_state(eval_atm_state_true, AtmosphericMismatchConfig())
    t_atm_true, l_path_true = surrogate(eval_atm_state_true)
    t_atm_assumed, l_path_assumed = surrogate(atm_state_assumed)

    l_sensor = sensor_radiance(adv_reflectance, t_atm_true, solar, l_path_true)
    r_rec = invert_to_reflectance(l_sensor, t_atm_assumed, solar, l_path_assumed)
    r_rec = torch.clamp(r_rec, 0.0, 1.0)

    with torch.no_grad():
        logits = classifier(r_rec.unsqueeze(-1))
        acc = (logits.argmax(1) == labels).float().mean().item()
    return acc


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    clean_spectra, labels, n_bands = load_data(device, N_SAMPLES)
    classifier = load_spectralformer_vit("Indian", device=device)
    classifier.eval()
    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=n_bands).to(device)
    surrogate.eval()
    solar = (torch.rand(n_bands, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.75).to(device)

    with torch.no_grad():
        clean_acc = (classifier(clean_spectra.unsqueeze(-1)).argmax(1) == labels).float().mean().item()
    print(f"clean accuracy (no attack): {clean_acc:.4f}\n")

    gen_atm_state = torch.tensor([GENERATION_ATM] * clean_spectra.shape[0], device=device)

    results: list[ASRResult] = []

    for epsilon in EPSILON_SWEEP:
        print(f"=== epsilon={epsilon} ===")

        rtaa_mismatch = SpectralFormerRTAAAttack(
            surrogate=surrogate, solar_irradiance=solar, epsilon=epsilon, step_size=epsilon / 5,
            n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(),
            mismatch_config=AtmosphericMismatchConfig(),
        )
        adv_rtaa_mismatch, _ = rtaa_mismatch.generate(
            classifier=classifier, clean_spectra=clean_spectra, labels=labels, atm_state=gen_atm_state
        )

        rtaa_ablation = SpectralFormerRTAAAttack(
            surrogate=surrogate, solar_irradiance=solar, epsilon=epsilon, step_size=epsilon / 5,
            n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(),
            mismatch_config=AtmosphericMismatchConfig.none(),
        )
        adv_rtaa_ablation, _ = rtaa_ablation.generate(
            classifier=classifier, clean_spectra=clean_spectra, labels=labels, atm_state=gen_atm_state
        )

        adv_pgd = pgd_attack(
            classifier, clean_spectra.unsqueeze(-1), labels,
            epsilon=epsilon, step_size=epsilon / 5, n_steps=N_STEPS,
        ).squeeze(-1)

        methods = {
            "RTAA (mismatch-aware)": adv_rtaa_mismatch,
            "RTAA (ablation, no mismatch)": adv_rtaa_ablation,
            "PGD (pixel-domain baseline)": adv_pgd,
        }

        for method_name, adv_spectra in methods.items():
            for cond_name, cond_values in EVAL_CONDITIONS.items():
                eval_atm_state = torch.tensor([cond_values] * clean_spectra.shape[0], device=device)
                adv_acc = simulate_and_evaluate(adv_spectra, labels, eval_atm_state, surrogate, solar, classifier)
                asr = 1.0 - adv_acc
                results.append(ASRResult(method_name, epsilon, cond_name, clean_acc, adv_acc, asr))
                print(f"  {method_name:32s} | {cond_name:15s} | adv_acc={adv_acc:.4f} | ASR={asr:.4f}")

    with open("asr_sweep_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print("\nSaved results to asr_sweep_results.json")


if __name__ == "__main__":
    main()
