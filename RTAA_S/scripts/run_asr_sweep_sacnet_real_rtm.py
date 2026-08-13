"""Reruns the SACNet ASR sweep (run_asr_sweep_sacnet.py) against the
real-libRadtran-trained RTM surrogate instead of the placeholder-physics one,
to check whether the ablation-controlled result and (best-case) scaffolding
finding survive the switch from analytic placeholder atmosphere to real
DISORT/REPTRAN radiative transfer. Everything else (classifier checkpoint,
sampling, seeds, evaluation conditions) is unchanged.
"""

from __future__ import annotations

import os

import run_asr_sweep_sacnet as base

base.RTM_CHECKPOINT = "checkpoints/rtm_surrogate_103bands_real.pt"

_OUT_PATH = "asr_sweep_sacnet_real_rtm_results.json"

if __name__ == "__main__":
    if os.path.exists("asr_sweep_sacnet_results.json"):
        os.rename("asr_sweep_sacnet_results.json", "asr_sweep_sacnet_results.json.protected_bak")
    try:
        base.main()
        os.replace("asr_sweep_sacnet_results.json", _OUT_PATH)
    finally:
        if os.path.exists("asr_sweep_sacnet_results.json.protected_bak"):
            os.rename("asr_sweep_sacnet_results.json.protected_bak", "asr_sweep_sacnet_results.json")
    print(f"Saved output to {_OUT_PATH}, original placeholder-physics results file restored/untouched")
