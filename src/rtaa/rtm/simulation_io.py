"""I/O contract for RTM simulation data used to train `RTMSurrogate`.

This is the schema your MODTRAN export script needs to produce. Once a .npz
file matching this schema exists, `train_surrogate.py` consumes it directly —
no other code changes needed.

Schema (single .npz file):
    atm_state       float32 (N, 3)        columns: [aerosol_optical_depth,
                                            water_vapor_cm, solar_zenith_deg]
    transmittance   float32 (N, n_bands)  T_atm(lambda) in [0, 1]
    path_radiance   float32 (N, n_bands)  L_path(lambda), same radiance units
                                           as the sensor radiance the
                                           classifier/attack pipeline uses
    wavelengths_nm  float32 (n_bands,)    band centers, must match the target
                                           dataset's band count (e.g. 103 for
                                           PaviaU, 200 for Indian Pines, 48 for
                                           Houston2018) — resample MODTRAN's
                                           native spectral resolution onto the
                                           sensor's band centers before saving

N is the number of simulated atmospheric-state samples (aim for a few
thousand covering the (tau, W, theta_s) ranges you need for the ASR sweep in
H_2 Phase 3, e.g. tau in [0.05, 0.5], W in [0.5, 5.0] cm, theta_s in [0, 70]
degrees).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RTMSimulationData:
    atm_state: np.ndarray  # (N, 3) float32
    transmittance: np.ndarray  # (N, n_bands) float32
    path_radiance: np.ndarray  # (N, n_bands) float32
    wavelengths_nm: np.ndarray  # (n_bands,) float32

    @property
    def n_samples(self) -> int:
        return self.atm_state.shape[0]

    @property
    def n_bands(self) -> int:
        return self.wavelengths_nm.shape[0]

    def validate(self) -> None:
        n, b = self.n_samples, self.n_bands
        if self.atm_state.shape != (n, 3):
            raise ValueError(f"atm_state expected shape ({n}, 3), got {self.atm_state.shape}")
        if self.transmittance.shape != (n, b):
            raise ValueError(f"transmittance expected shape ({n}, {b}), got {self.transmittance.shape}")
        if self.path_radiance.shape != (n, b):
            raise ValueError(f"path_radiance expected shape ({n}, {b}), got {self.path_radiance.shape}")
        if not np.isfinite(self.atm_state).all():
            raise ValueError("atm_state contains non-finite values")
        if not np.isfinite(self.transmittance).all():
            raise ValueError("transmittance contains non-finite values")
        if not np.isfinite(self.path_radiance).all():
            raise ValueError("path_radiance contains non-finite values")
        if (self.transmittance < 0).any() or (self.transmittance > 1).any():
            raise ValueError("transmittance must be in [0, 1]")
        if (self.path_radiance < 0).any():
            raise ValueError("path_radiance must be non-negative")

    def save(self, path: str | Path) -> None:
        self.validate()
        np.savez(
            path,
            atm_state=self.atm_state.astype(np.float32),
            transmittance=self.transmittance.astype(np.float32),
            path_radiance=self.path_radiance.astype(np.float32),
            wavelengths_nm=self.wavelengths_nm.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> RTMSimulationData:
        data = np.load(path)
        record = cls(
            atm_state=data["atm_state"],
            transmittance=data["transmittance"],
            path_radiance=data["path_radiance"],
            wavelengths_nm=data["wavelengths_nm"],
        )
        record.validate()
        return record
