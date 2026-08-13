"""ASR sweep on a third architecture — SACNet / PaviaU.

Same replicated protocol as SpectralFormer (`run_asr_sweep_replicated.py`)
and HybridSN (`run_asr_sweep_hybridsn.py`): RTAA (mismatch-aware) vs. its
own ablation vs. plain PGD, epsilon=0.01, 8 seeds, 4 evaluation atmospheres,
Bonferroni-corrected paired t-tests.

Structural difference from both prior targets: SACNet is a whole-scene
fully-convolutional network with a self-attention context module — it
consumes the entire (n_bands, H, W) PaviaU image in one forward pass, not
per-pixel or per-patch batches. There is therefore only one "sample" per
attack generation (the whole scene), so seed-to-seed variation here comes
from: which subset of the scene's ~40k held-out test pixels are included in
the attack's masked loss and evaluation (a fresh random subset per seed,
matching N_SAMPLES), plus independent solar-irradiance and mismatch-noise
draws per seed — the same sources of variation used in the other two
protocols, just without also varying which pixels' spectra get perturbed
(the whole scene is perturbed every time, as SACNet's architecture
requires; only the loss/eval mask changes).

PGD is reimplemented locally (`_pgd_attack_masked`) rather than reusing
`rtaa.attacks.baselines.pgd_attack`, because that function assumes an
unmasked per-sample cross-entropy loss — SACNet's masked whole-scene
setting needs `SACNetRTAAAttack._masked_cross_entropy` instead, so this
mirrors that exactly, changing only the loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch import Tensor, nn

from rtaa.attacks.sacnet_attack import (
    IGNORE_LABEL,
    PhysicalViabilityWeights,
    SACNetRTAAAttack,
)
from rtaa.models.sacnet import SACNET_REPO_DIR, load_sacnet
from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state
from rtaa.rtm.surrogate import RTMSurrogate

CHECKPOINT = "checkpoints/sacnet_paviau.pt"
RTM_CHECKPOINT = "checkpoints/rtm_surrogate_103bands.pt"
DATA_DIR = Path(SACNET_REPO_DIR) / "Data" / "PaviaU"
N_CLASSES = 9

GENERATION_ATM = [0.05, 0.8, 15.0]
EVAL_CONDITIONS = {
    "clear": [0.05, 0.8, 15.0],
    "moderate_haze": [0.2, 1.5, 30.0],
    "heavy_haze": [0.4, 2.5, 45.0],
    "extreme": [0.5, 4.0, 60.0],
}
EPSILON = 0.01
N_STEPS = 25
N_SAMPLES = 3000  # subset of ~40k test pixels used per seed
N_SEEDS = 8


def _pgd_attack_masked(
    model: nn.Module, clean_scene: Tensor, eval_labels: Tensor,
    epsilon: float, step_size: float, n_steps: int,
) -> Tensor:
    original = clean_scene.clone().detach()
    delta = torch.zeros_like(original)
    eval_labels_batched = eval_labels.unsqueeze(0)
    for _ in range(n_steps):
        adv = (original + delta).detach().requires_grad_(True)
        logits = model(adv.unsqueeze(0))
        loss = SACNetRTAAAttack._masked_cross_entropy(logits, eval_labels_batched)
        grad = torch.autograd.grad(loss, adv)[0]
        delta = delta + step_size * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)
    return torch.clamp(original + delta, 0.0, 1.0).detach()


def masked_accuracy(logits: Tensor, eval_labels: Tensor) -> float:
    pred = logits.argmax(1).squeeze(0)
    mask = eval_labels != IGNORE_LABEL
    return (pred[mask] == eval_labels[mask]).float().mean().item()


def simulate_and_evaluate_scene(
    adv_scene: Tensor, eval_labels: Tensor, eval_atm_state_true: Tensor,
    surrogate: RTMSurrogate, solar: Tensor, model: nn.Module,
) -> float:
    atm_state_assumed = perturb_atm_state(eval_atm_state_true, AtmosphericMismatchConfig())
    t_atm_true, l_path_true = surrogate(eval_atm_state_true)
    t_atm_assumed, l_path_assumed = surrogate(atm_state_assumed)
    t_atm_true_b = t_atm_true.squeeze(0)[:, None, None]
    l_path_true_b = l_path_true.squeeze(0)[:, None, None]
    t_atm_assumed_b = t_atm_assumed.squeeze(0)[:, None, None]
    l_path_assumed_b = l_path_assumed.squeeze(0)[:, None, None]
    solar_b = solar[:, None, None]

    l_sensor = sensor_radiance(adv_scene, t_atm_true_b, solar_b, l_path_true_b)
    r_rec = invert_to_reflectance(l_sensor, t_atm_assumed_b, solar_b, l_path_assumed_b)
    r_rec = torch.clamp(r_rec, 0.0, 1.0)

    with torch.no_grad():
        logits = model(r_rec.unsqueeze(0))
        acc = masked_accuracy(logits, eval_labels)
    return acc


def build_eval_labels(Y: np.ndarray, test_array: np.ndarray, h: int, w: int, sel: np.ndarray) -> np.ndarray:
    eval_labels = np.full(h * w, IGNORE_LABEL, dtype=np.int64)
    chosen = test_array[sel]
    eval_labels[chosen] = Y[chosen]
    return eval_labels.reshape(h, w)


def run_one_seed(
    seed: int, X: np.ndarray, Y: np.ndarray, test_array: np.ndarray,
    device: torch.device, surrogate: RTMSurrogate, model: nn.Module,
) -> dict:
    torch.manual_seed(seed)
    n_bands, h, w = X.shape
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(test_array), min(N_SAMPLES, len(test_array)), replace=False)
    eval_labels_np = build_eval_labels(Y, test_array, h, w, sel)

    clean_scene = torch.from_numpy(X).to(device)
    eval_labels = torch.from_numpy(eval_labels_np).to(device)
    solar = (torch.rand(n_bands, generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.75).to(device)
    gen_atm_state = torch.tensor([GENERATION_ATM], device=device)

    rtaa_mismatch = SACNetRTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig(),
    )
    adv_mismatch, _ = rtaa_mismatch.generate(classifier=model, clean_scene=clean_scene, eval_labels=eval_labels, atm_state=gen_atm_state)

    rtaa_ablation = SACNetRTAAAttack(
        surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
        n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig.none(),
    )
    adv_ablation, _ = rtaa_ablation.generate(classifier=model, clean_scene=clean_scene, eval_labels=eval_labels, atm_state=gen_atm_state)

    adv_pgd = _pgd_attack_masked(model, clean_scene, eval_labels, EPSILON, EPSILON / 5, N_STEPS)

    methods = {"mismatch": adv_mismatch, "ablation": adv_ablation, "pgd": adv_pgd}
    seed_result = {"seed": seed}
    for method_name, adv_scene in methods.items():
        for cond_name, cond_values in EVAL_CONDITIONS.items():
            eval_atm_state = torch.tensor([cond_values], device=device)
            adv_acc = simulate_and_evaluate_scene(adv_scene, eval_labels, eval_atm_state, surrogate, solar, model)
            seed_result[f"{method_name}__{cond_name}"] = 1.0 - adv_acc
    return seed_result


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X = np.load(DATA_DIR / "X.npy").astype(np.float32)
    Y = np.load(DATA_DIR / "Y.npy")
    test_array = np.load(DATA_DIR / "test_array.npy")
    n_bands = X.shape[0]
    print(f"Scene shape {X.shape}, {len(test_array)} held-out test pixels")

    model = load_sacnet(n_bands=n_bands, n_classes=N_CLASSES, device=str(device))
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=n_bands).to(device)
    surrogate.eval()

    all_results = []
    for seed in range(N_SEEDS):
        print(f"=== seed {seed}/{N_SEEDS-1} ===")
        result = run_one_seed(seed, X, Y, test_array, device, surrogate, model)
        all_results.append(result)
        for cond_name in EVAL_CONDITIONS:
            m, a, p = result[f"mismatch__{cond_name}"], result[f"ablation__{cond_name}"], result[f"pgd__{cond_name}"]
            print(f"  {cond_name:15s} ASR: mismatch={m:.4f}  ablation={a:.4f}  pgd={p:.4f}  (mismatch-pgd={m-p:+.4f})")

    with open("asr_sweep_sacnet_results.json", "w") as f:
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

    print("\nSaved per-seed results to asr_sweep_sacnet_results.json")


if __name__ == "__main__":
    main()
