"""Same gradient-level diagnostic as run_gradient_diagnostic.py, applied to
SACNet -- the one target with a POSITIVE scaffolding term (+0.0224,
Table `tab:decomposition`) -- to test whether gradient misalignment between
RTAA's and PGD's paths at delta=0 tracks the SIGN of the scaffolding term
across architectures, not just its magnitude on HybridSN.

If low cosine similarity is specifically a HybridSN/PCA-chain artifact, this
should not reproduce the near-orthogonal (~0.10) gradient alignment found
there. If it does reproduce, gradient misalignment is not, on its own,
sufficient to explain scaffolding cost sign -- something else (how the
architecture's own decision surface responds to a given direction) has to
be doing the rest of the work.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from run_asr_sweep_sacnet import (
    DATA_DIR,
    GENERATION_ATM,
    N_CLASSES,
    N_SAMPLES,
    RTM_CHECKPOINT,
    build_eval_labels,
)

from rtaa.models.sacnet import load_sacnet
from rtaa.rtm.forward_model import invert_to_reflectance, sensor_radiance
from rtaa.rtm.mismatch import AtmosphericMismatchConfig, perturb_atm_state
from rtaa.rtm.surrogate import RTMSurrogate

CHECKPOINT = "checkpoints/sacnet_paviau.pt"
N_SEEDS = 8


def concentration(grad: torch.Tensor) -> float:
    flat = grad.abs().flatten()
    total = flat.sum().item()
    k = max(1, int(0.10 * flat.numel()))
    return torch.topk(flat, k).values.sum().item() / total


def one_seed(seed, X, Y, test_array, device, surrogate, model):
    torch.manual_seed(seed)
    n_bands, h, w = X.shape
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(test_array), min(N_SAMPLES, len(test_array)), replace=False)
    eval_labels_np = build_eval_labels(Y, test_array, h, w, sel)

    clean_scene = torch.from_numpy(X).to(device)
    eval_labels = torch.from_numpy(eval_labels_np).to(device)
    solar = (torch.rand(n_bands, generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.75).to(device)
    gen_atm_state = torch.tensor([GENERATION_ATM], device=device)
    eval_labels_batched = eval_labels.unsqueeze(0)

    def masked_ce(logits):
        n, c, hh, ww = logits.shape
        mask = eval_labels_batched != 255
        logits_flat = logits.permute(0, 2, 3, 1)[mask.view(n, hh, ww)].view(-1, c)
        labels_flat = eval_labels_batched[mask]
        return torch.nn.functional.cross_entropy(logits_flat, labels_flat)

    # --- PGD's gradient: delta=0, direct to classifier ---
    delta_pgd = torch.zeros_like(clean_scene, requires_grad=True)
    logits_pgd = model((clean_scene + delta_pgd).unsqueeze(0))
    loss_pgd = masked_ce(logits_pgd)
    grad_pgd = torch.autograd.grad(loss_pgd, delta_pgd)[0]

    # --- RTAA's gradient: delta=0, through RTM forward -> RTM inverse -> classifier ---
    delta_rtaa = torch.zeros_like(clean_scene, requires_grad=True)
    mismatch_config = AtmosphericMismatchConfig()
    atm_state_assumed = perturb_atm_state(gen_atm_state, mismatch_config)
    t_atm_true, l_path_true = surrogate(gen_atm_state)
    t_atm_assumed, l_path_assumed = surrogate(atm_state_assumed)
    t_atm_true_b, l_path_true_b = t_atm_true.squeeze(0)[:, None, None], l_path_true.squeeze(0)[:, None, None]
    t_atm_assumed_b, l_path_assumed_b = t_atm_assumed.squeeze(0)[:, None, None], l_path_assumed.squeeze(0)[:, None, None]
    solar_b = solar[:, None, None]

    r_adv = torch.clamp(clean_scene + delta_rtaa, 0.0, 1.0)
    l_sensor = sensor_radiance(r_adv, t_atm_true_b, solar_b, l_path_true_b)
    r_rec = invert_to_reflectance(l_sensor, t_atm_assumed_b, solar_b, l_path_assumed_b)
    logits_rtaa = model(r_rec.unsqueeze(0))
    loss_rtaa = masked_ce(logits_rtaa)
    grad_rtaa = torch.autograd.grad(loss_rtaa, delta_rtaa)[0]

    cos_sim = torch.nn.functional.cosine_similarity(grad_pgd.flatten(), grad_rtaa.flatten(), dim=0).item()

    return {
        "seed": seed,
        "grad_norm_pgd": grad_pgd.norm().item(),
        "grad_norm_rtaa": grad_rtaa.norm().item(),
        "concentration_pgd": concentration(grad_pgd),
        "concentration_rtaa": concentration(grad_rtaa),
        "cosine_similarity": cos_sim,
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = np.load(DATA_DIR / "X.npy").astype(np.float32)
    Y = np.load(DATA_DIR / "Y.npy")
    test_array = np.load(DATA_DIR / "test_array.npy")
    n_bands = X.shape[0]

    model = load_sacnet(n_bands=n_bands, n_classes=N_CLASSES, device=str(device))
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=n_bands).to(device)
    surrogate.eval()

    results = []
    for seed in range(N_SEEDS):
        r = one_seed(seed, X, Y, test_array, device, surrogate, model)
        results.append(r)
        print(f"seed {seed}: ||grad||_PGD={r['grad_norm_pgd']:.6f}  ||grad||_RTAA={r['grad_norm_rtaa']:.6f}  "
              f"ratio={r['grad_norm_rtaa']/r['grad_norm_pgd']:.4f}  cos_sim={r['cosine_similarity']:.4f}  "
              f"conc_PGD={r['concentration_pgd']:.4f}  conc_RTAA={r['concentration_rtaa']:.4f}")

    print(f"\n=== Summary across {N_SEEDS} seeds (SACNet) ===")
    for key in ["grad_norm_pgd", "grad_norm_rtaa", "concentration_pgd", "concentration_rtaa", "cosine_similarity"]:
        vals = np.array([r[key] for r in results])
        print(f"  {key:20s} mean={vals.mean():.5f}  std={vals.std(ddof=1):.5f}")

    with open("sacnet_gradient_diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to sacnet_gradient_diagnostic_results.json")


if __name__ == "__main__":
    main()
