"""Generates real-RTM (atm_state -> T_atm, L_path) training data for
`RTMSurrogate`, using libRadtran's `uvspec` (DISORT solver, REPTRAN
correlated-k gas absorption) in place of the analytic placeholder physics in
`placeholder_physics.py`. Output matches the schema in `simulation_io.py`
exactly, so `train_surrogate.py --data <this script's output>` consumes it
with no other code changes.

Requires: `conda activate libradtran` (see PUBLICATION_ROADMAP.md /
PAPER_ROADMAP.md for the setup — installed via `conda create -n libradtran
-c conda-forge rubin-libradtran`).

--- Method ---
uvspec computes physical top-of-atmosphere radiance L_TOA(lambda; A) for a
Lambertian surface of a *given* albedo A, not T_atm/L_path directly. We
recover both by running uvspec at several albedos per atmospheric state and
fitting the linear model implied by this project's own forward model
(`rtaa.rtm.forward_model.sensor_radiance`):

    L_TOA(lambda; A) = L_path(lambda) + T_atm(lambda) * E_sun(lambda) * A

per wavelength, via least squares across the albedo sweep. This ignores the
atmospheric spherical-albedo (surface-atmosphere multiple-reflection) term —
consistent with (not an approximation error relative to) the project's own
linear radiative-transfer model, which also omits it.

`E_sun(lambda)` in that equation is defined as the extraterrestrial solar
irradiance (Kurudz spectrum, the same file `uvspec` uses) scaled by
cos(solar_zenith)/pi, i.e. the standard "surface-normalized" solar term.
Dividing the fitted slope/intercept through by this quantity turns both
T_atm and L_path into dimensionless numbers on the same [0, ~1] scale as
reflectance — T_atm as a true two-way transmittance fraction, L_path as the
"atmospheric intrinsic reflectance" (same convention used in 6S/FLAASH-style
atmospheric correction). This is what makes the output units-compatible with
`sensor_radiance`'s R in [0, 1] and the pipeline's own (arbitrary-magnitude,
~[0.75, 1.25]) `solar_irradiance` tensor: T_atm stays physically meaningful
regardless of what solar-irradiance scale a downstream script picks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.rtaa.rtm.simulation_io import RTMSimulationData

SENSOR_PRESETS = {
    # name: (n_bands, wl_lo_nm, wl_hi_nm)
    "PaviaU": (103, 430, 860),
    "Houston2018": (48, 380, 1050),
}

# AVIRIS's "corrected" Indian Pines product (the 200-band cube this project
# trains against) removes bands falling in the two deep atmospheric
# water-vapor absorption windows (~1350-1465nm, ~1790-1960nm) — those bands
# are unusable on any real sensor, which is why the public 200-band release
# excludes them rather than shipping them as near-zero-signal noise. Neither
# local Indian Pines .mat file (SpectralFormer's or the SAFER mirror's)
# bundles the literal per-band calibration used to build that release, so
# this grid is NOT guaranteed to match any specific paper's exact 200 band
# centers — but spacing samples only across the sensor's genuinely-usable
# windows (skipping the blackout zones entirely) is still a materially
# better approximation than a blind linspace(400, 2500, 200), which would
# assign real bands to wavelengths where transmittance is physically ~0 and
# no real classifier ever sees data.
INDIAN_PINES_USABLE_WINDOWS_NM = [(400.0, 1350.0), (1465.0, 1790.0), (1960.0, 2500.0)]
INDIAN_PINES_N_BANDS = 200


def _indian_pines_wavelength_grid() -> np.ndarray:
    window_widths = [hi - lo for lo, hi in INDIAN_PINES_USABLE_WINDOWS_NM]
    total_width = sum(window_widths)
    bands_per_window = [round(INDIAN_PINES_N_BANDS * w / total_width) for w in window_widths]
    bands_per_window[-1] += INDIAN_PINES_N_BANDS - sum(bands_per_window)  # fix rounding drift
    grids = [
        np.linspace(lo, hi, n, endpoint=False)
        for (lo, hi), n in zip(INDIAN_PINES_USABLE_WINDOWS_NM, bands_per_window)
    ]
    return np.concatenate(grids).astype(np.float32)

DEFAULT_ALBEDOS = (0.0, 0.15, 0.3, 0.5, 0.7)


def _find_data_path() -> str:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    candidate = os.path.join(conda_prefix, "share", "libRadtran", "data")
    if conda_prefix and os.path.isdir(candidate):
        return candidate
    raise RuntimeError(
        "Could not locate libRadtran data directory. Activate the libradtran "
        "conda env first (`conda activate libradtran`), or pass --data-path explicitly."
    )


def _load_solar_flux(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(data_path, "solar_flux", "kurudz_1.0nm.dat")
    arr = np.loadtxt(path, comments="#")
    return arr[:, 0], arr[:, 1]  # wavelength_nm, irradiance_mW_m2_nm


def _run_uvspec(
    uvspec_bin: str, data_path: str, tau550: float, water_vapor_mm: float,
    sza_deg: float, albedo: float, wl_lo: float, wl_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (wavelength_nm, toa_nadir_radiance) at the given albedo."""
    inp = f"""\
data_files_path {data_path}/
atmosphere_file {data_path}/atmmod/afglus.dat
source solar {data_path}/solar_flux/kurudz_1.0nm.dat
mol_abs_param reptran coarse
mol_modify H2O {water_vapor_mm} MM
aerosol_default
aerosol_modify tau550 set {tau550}
rte_solver disort
sza {sza_deg}
zout toa
umu 1
phi 0
albedo {albedo}
wavelength {wl_lo} {wl_hi}
output_user lambda edir uu
quiet
"""
    result = subprocess.run(
        [uvspec_bin], input=inp, capture_output=True, text=True, check=True,
    )
    rows = np.array([line.split() for line in result.stdout.splitlines() if line.strip()], dtype=np.float64)
    return rows[:, 0], rows[:, 2]  # lambda, uu (edir is column 1, unused here)


def simulate_one_state(
    uvspec_bin: str, data_path: str, tau550: float, water_vapor_mm: float,
    sza_deg: float, albedos: tuple[float, ...], wl_lo: float, wl_hi: float,
    target_wavelengths_nm: np.ndarray, solar_wl: np.ndarray, solar_irr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (transmittance, path_radiance) resampled onto target_wavelengths_nm."""
    native_wl = None
    radiances = []
    for albedo in albedos:
        wl, uu = _run_uvspec(uvspec_bin, data_path, tau550, water_vapor_mm, sza_deg, albedo, wl_lo, wl_hi)
        if native_wl is None:
            native_wl = wl
        radiances.append(uu)
    radiance_matrix = np.stack(radiances, axis=0)  # (n_albedo, n_wl_native)

    albedo_arr = np.asarray(albedos)
    # per-wavelength least-squares linear fit: uu(A) = slope * A + intercept
    design = np.stack([albedo_arr, np.ones_like(albedo_arr)], axis=1)  # (n_albedo, 2)
    coeffs, *_ = np.linalg.lstsq(design, radiance_matrix, rcond=None)  # (2, n_wl_native)
    slope, intercept = coeffs[0], coeffs[1]

    e_sun_native = np.interp(native_wl, solar_wl, solar_irr)
    norm = e_sun_native * np.cos(np.deg2rad(sza_deg)) / np.pi
    norm = np.clip(norm, 1e-8, None)

    t_atm_native = np.clip(slope / norm, 0.0, 1.0)
    l_path_native = np.clip(intercept / norm, 0.0, None)

    t_atm = np.interp(target_wavelengths_nm, native_wl, t_atm_native)
    l_path = np.interp(target_wavelengths_nm, native_wl, l_path_native)
    return t_atm.astype(np.float32), l_path.astype(np.float32)


def _worker(args: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    (idx, uvspec_bin, data_path, tau, water_vapor_cm, sza_deg, albedos,
     wl_lo, wl_hi, target_wavelengths_nm, solar_wl, solar_irr) = args
    water_vapor_mm = water_vapor_cm * 10.0  # cm -> mm precipitable water
    t_atm, l_path = simulate_one_state(
        uvspec_bin, data_path, tau, water_vapor_mm, sza_deg, albedos,
        wl_lo, wl_hi, target_wavelengths_nm, solar_wl, solar_irr,
    )
    return idx, t_atm, l_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor", choices=sorted(set(SENSOR_PRESETS) | {"IndianPines"}), default="IndianPines")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--tau-range", type=float, nargs=2, default=(0.05, 0.5))
    parser.add_argument("--water-vapor-range", type=float, nargs=2, default=(0.5, 5.0), help="cm, matches AtmosphericMismatchConfig convention")
    parser.add_argument("--solar-zenith-range", type=float, nargs=2, default=(0.0, 70.0))
    parser.add_argument("--albedos", type=float, nargs="+", default=list(DEFAULT_ALBEDOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uvspec-bin", type=str, default="uvspec")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--n-jobs", type=int, default=min(32, os.cpu_count() or 4))
    parser.add_argument("--out", type=str, default="libradtran_rtm_data.npz")
    args = parser.parse_args()

    data_path = args.data_path or _find_data_path()
    if args.sensor == "IndianPines":
        n_bands = INDIAN_PINES_N_BANDS
        wl_lo, wl_hi = INDIAN_PINES_USABLE_WINDOWS_NM[0][0], INDIAN_PINES_USABLE_WINDOWS_NM[-1][1]
        target_wavelengths_nm = _indian_pines_wavelength_grid()
    else:
        n_bands, wl_lo, wl_hi = SENSOR_PRESETS[args.sensor]
        target_wavelengths_nm = np.linspace(wl_lo, wl_hi, n_bands).astype(np.float32)
    solar_wl, solar_irr = _load_solar_flux(data_path)

    rng = np.random.default_rng(args.seed)
    tau = rng.uniform(*args.tau_range, size=args.n_samples)
    water_vapor_cm = rng.uniform(*args.water_vapor_range, size=args.n_samples)
    sza_deg = rng.uniform(*args.solar_zenith_range, size=args.n_samples)
    atm_state = np.stack([tau, water_vapor_cm, sza_deg], axis=1).astype(np.float32)

    transmittance = np.zeros((args.n_samples, n_bands), dtype=np.float32)
    path_radiance = np.zeros((args.n_samples, n_bands), dtype=np.float32)

    jobs = [
        (i, args.uvspec_bin, data_path, float(tau[i]), float(water_vapor_cm[i]), float(sza_deg[i]),
         tuple(args.albedos), wl_lo, wl_hi, target_wavelengths_nm, solar_wl, solar_irr)
        for i in range(args.n_samples)
    ]

    n_done = 0
    with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for future in as_completed(futures):
            idx, t_atm, l_path = future.result()
            transmittance[idx] = t_atm
            path_radiance[idx] = l_path
            n_done += 1
            if n_done % 10 == 0 or n_done == args.n_samples:
                print(f"  {n_done}/{args.n_samples} atmospheric states simulated")

    data = RTMSimulationData(atm_state, transmittance, path_radiance, target_wavelengths_nm)
    data.save(args.out)
    print(f"Saved {args.n_samples} samples ({n_bands} bands, {args.sensor} preset) to {args.out}")
    print(f"Train the surrogate with: uv run python -m rtaa.rtm.train_surrogate --data {args.out} --out rtm_surrogate_real.pt")


if __name__ == "__main__":
    main()
