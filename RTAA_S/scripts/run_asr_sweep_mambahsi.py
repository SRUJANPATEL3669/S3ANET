"""ASR sweep on a fifth architecture — MambaHSI / PaviaU: the first
image-level HSI classifier built on a real Mamba/state-space model, and
architecturally the furthest departure yet from SpectralFormer's attention
or HybridSN's plain convolution.

Same replicated protocol as SACNet/S3ANet (whole-scene attack via
`SACNetRTAAAttack`, reused unchanged through the `UpsampledMambaHSI`
wrapper — see `rtaa.models.mambahsi`), epsilon=0.01, 8 seeds, 4 evaluation
atmospheres, Bonferroni-corrected paired t-tests.

Data pipeline is MambaHSI's own, not SACNet/S3ANet's — it does not ship an
X.npy/Y.npy/test_array.npy convention. This script replicates the exact
sequence `scripts/train_mambahsi_paviau.py` uses to build train/val/test
splits (`data_load_operate.load_data` -> `ImageStretching` ->
`data_load_operate.sampling([0.1, 0.01], [30, 10], ..., Flag=1)`, seed=0
before the numpy-RNG-dependent `sampling` call) so the reconstructed test
set actually matches what the checkpoint was evaluated against, not an
independently-drawn and potentially leaky split.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from scipy import stats
from torch import Tensor, nn

from rtaa.attacks.sacnet_attack import (
    IGNORE_LABEL,
    PhysicalViabilityWeights,
    SACNetRTAAAttack,
)
from rtaa.models.mambahsi import MAMBAHSI_REPO_DIR, UpsampledMambaHSI, load_mambahsi
from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state
from rtaa.rtm.surrogate import RTMSurrogate

sys.path.insert(0, MAMBAHSI_REPO_DIR)
from utils import data_load_operate
from utils.HSICommonUtils import ImageStretching

CHECKPOINT = "checkpoints/mambahsi_paviau.pt"
RTM_CHECKPOINT = "checkpoints/rtm_surrogate_103bands.pt"
HIDDEN_DIM = 128
SPLIT_SEED = 0  # matches train_mambahsi_paviau.py's default

GENERATION_ATM = [0.05, 0.8, 15.0]
EVAL_CONDITIONS = {
    "clear": [0.05, 0.8, 15.0],
    "moderate_haze": [0.2, 1.5, 30.0],
    "heavy_haze": [0.4, 2.5, 45.0],
    "extreme": [0.5, 4.0, 60.0],
}
EPSILON = 0.01
N_STEPS = 25
N_SAMPLES = 3000
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


def build_full_test_eval_labels(device: torch.device) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Replicates train_mambahsi_paviau.py's exact data/split pipeline to
    recover the same test set the checkpoint was evaluated against.
    Returns (clean_scene_np (n_bands,H,W) in [0,1], full_test_eval_labels
    (H,W) with IGNORE_LABEL elsewhere, n_bands, h, w)."""
    np.random.seed(SPLIT_SEED)
    data, gt = data_load_operate.load_data("UP", f"{MAMBAHSI_REPO_DIR}/data")
    height, width, channels = data.shape
    gt_reshape = gt.reshape(-1)
    class_count = int(np.max(np.unique(gt)))

    img = ImageStretching(data)  # uint8 [0,255], percentile-stretched
    clean_scene_np = (np.array(img).astype(np.float32) / 255.0).transpose(2, 0, 1)  # (n_bands,H,W) in [0,1]

    train_idx, val_idx, test_idx, _ = data_load_operate.sampling(
        [0.1, 0.01], [30, 10], gt_reshape, class_count, 1
    )
    _train_labels, _val_labels, test_labels = data_load_operate.generate_image_iter(
        data, height, width, gt_reshape, (train_idx, val_idx, test_idx)
    )
    test_labels_np = test_labels.numpy().astype(np.int64)  # (H,W), -1 = excluded
    full_test_eval_labels = np.where(test_labels_np == -1, IGNORE_LABEL, test_labels_np)
    return clean_scene_np, full_test_eval_labels, channels, height, width


def run_one_seed(
    seed: int, clean_scene_np: np.ndarray, full_test_eval_labels: np.ndarray,
    n_bands: int, h: int, w: int, device: torch.device, surrogate: RTMSurrogate, model: nn.Module,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    test_rows, test_cols = np.nonzero(full_test_eval_labels != IGNORE_LABEL)
    sel = rng.choice(len(test_rows), min(N_SAMPLES, len(test_rows)), replace=False)

    eval_labels_np = np.full((h, w), IGNORE_LABEL, dtype=np.int64)
    rows, cols = test_rows[sel], test_cols[sel]
    eval_labels_np[rows, cols] = full_test_eval_labels[rows, cols]

    clean_scene = torch.from_numpy(clean_scene_np).to(device)
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

    clean_scene_np, full_test_eval_labels, n_bands, h, w = build_full_test_eval_labels(device)
    n_test = int((full_test_eval_labels != IGNORE_LABEL).sum())
    print(f"Scene shape ({n_bands},{h},{w}), {n_test} held-out test pixels")

    base_model = load_mambahsi(n_bands=n_bands, n_classes=9, hidden_dim=HIDDEN_DIM, device=device)
    base_model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model = UpsampledMambaHSI(base_model).to(device)
    model.eval()

    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=n_bands).to(device)
    surrogate.eval()

    all_results = []
    for seed in range(N_SEEDS):
        print(f"=== seed {seed}/{N_SEEDS-1} ===")
        result = run_one_seed(seed, clean_scene_np, full_test_eval_labels, n_bands, h, w, device, surrogate, model)
        all_results.append(result)
        for cond_name in EVAL_CONDITIONS:
            m, a, p = result[f"mismatch__{cond_name}"], result[f"ablation__{cond_name}"], result[f"pgd__{cond_name}"]
            print(f"  {cond_name:15s} ASR: mismatch={m:.4f}  ablation={a:.4f}  pgd={p:.4f}  (mismatch-pgd={m-p:+.4f})")

    with open("asr_sweep_mambahsi_results.json", "w") as f:
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

    print("\nSaved per-seed results to asr_sweep_mambahsi_results.json")


if __name__ == "__main__":
    main()
