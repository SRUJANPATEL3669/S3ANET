"""Reruns the SpectralFormer patch-wise ASR sweep against the
real-libRadtran-trained RTM surrogate instead of the placeholder-physics
one. See run_asr_sweep_hybridsn_real_rtm.py for rationale.
"""

from __future__ import annotations

import run_asr_sweep_spectralformer_patchwise as base
from _real_rtm_helper import run_protected

if __name__ == "__main__":
    run_protected(
        base,
        checkpoint_override="checkpoints/rtm_surrogate_200bands_real.pt",
        base_out_path="asr_sweep_spectralformer_patchwise_results.json",
        real_out_path="asr_sweep_spectralformer_patchwise_real_rtm_results.json",
    )
