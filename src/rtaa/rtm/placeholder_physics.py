"""Analytic placeholder atmosphere, for validating the surrogate training
pipeline before real 6S/MODTRAN simulation data is available.

THIS IS NOT A RADIATIVE TRANSFER MODEL. It is a physically-motivated but
deliberately crude closed-form stand-in (Rayleigh + Angstrom-law aerosol
extinction, single-scattering path-radiance approximation) used only to
generate (atm_state -> T_atm, L_path) pairs with roughly the right shape and
behavior, so `train_surrogate.py` and `RTMSurrogate` can be exercised
end-to-end. Any surrogate trained on this data is a pipeline smoke test, not
a scientifically usable RTM surrogate — retrain against real MODTRAN output
once available (see `simulation_io.py` for the expected schema).
"""

from __future__ import annotations

import numpy as np

# Sensor wavelength presets (matching generate_libradtran_rtm_data.py)
SENSOR_WAVELENGTH_PRESETS = {
    103: ("PaviaU", 430.0, 860.0),      # ROSIS sensor
    204: ("Salinas", 400.0, 2500.0),     # AVIRIS full
}

_INDIAN_PINES_USABLE_WINDOWS_NM = [(400.0, 1350.0), (1465.0, 1790.0), (1960.0, 2500.0)]

def _indian_pines_wavelength_grid() -> np.ndarray:
    """200 bands across AVIRIS usable windows, skipping water-vapor absorption."""
    window_widths = [hi - lo for lo, hi in _INDIAN_PINES_USABLE_WINDOWS_NM]
    total_width = sum(window_widths)
    bands_per_window = [round(200 * w / total_width) for w in window_widths]
    bands_per_window[-1] += 200 - sum(bands_per_window)  # fix rounding drift
    grids = [
        np.linspace(lo, hi, n, endpoint=False)
        for (lo, hi), n in zip(_INDIAN_PINES_USABLE_WINDOWS_NM, bands_per_window)
    ]
    return np.concatenate(grids).astype(np.float32)

def sensor_wavelengths(n_bands: int) -> np.ndarray:
    """Returns the appropriate wavelength grid for a given band count."""
    if n_bands == 200:
        return _indian_pines_wavelength_grid()
    if n_bands in SENSOR_WAVELENGTH_PRESETS:
        _, lo, hi = SENSOR_WAVELENGTH_PRESETS[n_bands]
        return np.linspace(lo, hi, n_bands).astype(np.float32)
    return np.linspace(400, 2500, n_bands).astype(np.float32)

def generate_synthetic_rtm_pairs(
    n_samples: int = 5000,
    n_bands: int = 103,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Notebook-friendly wrapper: picks sensor-appropriate wavelengths,
    then delegates to generate_placeholder_dataset."""
    wavelengths_nm = sensor_wavelengths(n_bands)
    return generate_placeholder_dataset(wavelengths_nm, n_samples=n_samples, seed=seed)


def generate_placeholder_dataset(
    wavelengths_nm: np.ndarray,
    n_samples: int = 4000,
    tau_range: tuple[float, float] = (0.05, 0.5),
    water_vapor_range: tuple[float, float] = (0.5, 5.0),
    solar_zenith_range_deg: tuple[float, float] = (0.0, 70.0),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (atm_state, transmittance, path_radiance).

    atm_state columns: [aerosol_optical_depth, water_vapor_cm, solar_zenith_deg].
    """
    rng = np.random.default_rng(seed)
    tau = rng.uniform(*tau_range, size=n_samples).astype(np.float32)
    water_vapor = rng.uniform(*water_vapor_range, size=n_samples).astype(np.float32)
    theta_s_deg = rng.uniform(*solar_zenith_range_deg, size=n_samples).astype(np.float32)
    atm_state = np.stack([tau, water_vapor, theta_s_deg], axis=1)

    mu0 = np.cos(np.deg2rad(theta_s_deg))  # (N,)
    lam_um = (wavelengths_nm / 1000.0).astype(np.float32)  # (n_bands,)

    # Rayleigh optical depth ~ lambda^-4 (Angstrom exponent ~4), scaled arbitrarily.
    tau_rayleigh = 0.008 * lam_um[None, :] ** (-4)  # (1, n_bands)

    # Aerosol optical depth via Angstrom power law (exponent ~1.3), anchored at tau(550nm).
    angstrom_exponent = 1.3
    tau_aerosol = tau[:, None] * (lam_um[None, :] / 0.55) ** (-angstrom_exponent)

    # Water vapor absorption: crude Gaussian absorption bands around 940nm and 1140nm.
    def water_absorption_depth(center_nm: float, width_nm: float) -> np.ndarray:
        return np.exp(-0.5 * ((wavelengths_nm - center_nm) / width_nm) ** 2)

    water_band = water_absorption_depth(940.0, 25.0) + 0.6 * water_absorption_depth(1140.0, 30.0)
    tau_water = water_vapor[:, None] * water_band[None, :] * 0.15

    tau_total = tau_rayleigh + tau_aerosol + tau_water  # (N, n_bands)
    two_way_path = 1.0 / np.clip(mu0[:, None], 0.05, None) + 1.0  # sun-to-surface-to-sensor path
    transmittance = np.exp(-tau_total * two_way_path).astype(np.float32)
    transmittance = np.clip(transmittance, 0.0, 1.0)

    # Single-scattering path radiance approximation, arbitrary units consistent
    # with `sensor_radiance`'s (T_atm * R * E_sun + L_path) formulation.
    single_scatter_albedo = 0.9
    path_radiance = (
        single_scatter_albedo * (1.0 - transmittance) * (0.3 + 0.2 * mu0[:, None])
    ).astype(np.float32)

    return atm_state, transmittance, path_radiance
