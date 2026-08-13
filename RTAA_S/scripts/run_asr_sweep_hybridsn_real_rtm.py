"""Reruns the HybridSN ASR sweep (run_asr_sweep_hybridsn.py) against the
real-libRadtran-trained RTM surrogate instead of the placeholder-physics one,
to check whether the ablation-controlled result and scaffolding-cost finding
survive the switch from analytic placeholder atmosphere to real DISORT/
REPTRAN radiative transfer. Everything else (classifier checkpoint, PCA,
sampling, seeds, evaluation conditions) is unchanged.
"""

from __future__ import annotations

import os

import run_asr_sweep_hybridsn as base

base.RTM_CHECKPOINT = "checkpoints/rtm_surrogate_200bands_real.pt"

_OUT_PATH = "asr_sweep_hybridsn_real_rtm_results.json"

if __name__ == "__main__":
    if os.path.exists("asr_sweep_hybridsn_results.json"):
        os.rename("asr_sweep_hybridsn_results.json", "asr_sweep_hybridsn_results.json.protected_bak")
    try:
        base.main()
        os.replace("asr_sweep_hybridsn_results.json", _OUT_PATH)
    finally:
        if os.path.exists("asr_sweep_hybridsn_results.json.protected_bak"):
            os.rename("asr_sweep_hybridsn_results.json.protected_bak", "asr_sweep_hybridsn_results.json")
    print(f"Saved output to {_OUT_PATH}, original placeholder-physics results file restored/untouched")
