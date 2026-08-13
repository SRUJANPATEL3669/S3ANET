"""Calibrates C&W's `c` hyperparameter so its resulting mean L_inf
perturbation matches RTAA's fixed epsilon=0.01 budget, before rerunning the
budget-matched comparison (PUBLICATION_ROADMAP.md: the c=10 run used 3.4x
RTAA's budget, making that comparison uninformative).

Binary search over log(c): C&W's perturbation grows monotonically with c
(stronger misclassification pressure trades off against a larger L2/Linf
penalty), so a plain bisection on log(c) converges quickly. Uses the same
classifier/data/CW_STEPS as the full comparison, but a smaller sample count
for speed — c is a global scalar hyperparameter, not dependent on which
particular pixels are sampled.
"""

from __future__ import annotations

import torch
from run_asr_sweep import (  # noqa: F401 (N_STEPS unused here, kept for parity)
    N_STEPS,
    RTM_CHECKPOINT,
    load_data,
)
from run_asr_sweep_cw import CW_LR, CW_STEPS

from rtaa.attacks.baselines import cw_attack
from rtaa.models.spectralformer import load_spectralformer_vit

TARGET_LINF = 0.01
N_CALIB_SAMPLES = 100
N_ITERS = 12


def mean_linf(classifier, clean_spectra, labels, c: float) -> float:
    adv = cw_attack(classifier, clean_spectra.unsqueeze(-1), labels, n_steps=CW_STEPS, lr=CW_LR, c=c).squeeze(-1)
    return (adv - clean_spectra).abs().max(dim=1).values.mean().item()


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    classifier = load_spectralformer_vit("Indian", device=device)
    classifier.eval()
    clean_spectra, labels, _n_bands = load_data(device, N_CALIB_SAMPLES, seed=0)

    lo, hi = 0.001, 0.01
    linf_lo = mean_linf(classifier, clean_spectra, labels, lo)
    linf_hi = mean_linf(classifier, clean_spectra, labels, hi)
    print(f"c={lo:.4f} -> linf={linf_lo:.4f}")
    print(f"c={hi:.4f} -> linf={linf_hi:.4f}")
    if not (linf_lo <= TARGET_LINF <= linf_hi):
        print(f"WARNING: target {TARGET_LINF} not bracketed by [{lo}, {hi}]; search may not converge cleanly.")

    for i in range(N_ITERS):
        mid = (lo * hi) ** 0.5  # geometric mean (bisection in log-space)
        linf_mid = mean_linf(classifier, clean_spectra, labels, mid)
        print(f"iter {i}: c={mid:.5f} -> linf={linf_mid:.4f}")
        if linf_mid > TARGET_LINF:
            hi = mid
        else:
            lo = mid

    final_c = (lo * hi) ** 0.5
    final_linf = mean_linf(classifier, clean_spectra, labels, final_c)
    print(f"\nCalibrated c = {final_c:.5f}, resulting mean L_inf = {final_linf:.4f} (target {TARGET_LINF})")


if __name__ == "__main__":
    main()
