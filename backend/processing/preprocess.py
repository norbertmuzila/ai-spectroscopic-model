"""
Spectral preprocessing.

Order matters here, and getting it wrong silently destroys the features the
identification depends on:

    raw counts
      -> dark subtraction              (removes detector offset + thermal noise)
      -> nonlinearity correction       (USB4000 ADC is not perfectly linear)
      -> reflectance                   (ratio against a Spectralon white standard)
      -> despike                       (cosmic rays, hot pixels)
      -> resample to common grid       (so library and sample are comparable)
      -> Savitzky-Golay smoothing      (denoise WITHOUT flattening bands)
      -> continuum removal             (isolates absorption features from albedo)

Continuum removal is the single most important step for mineral identification.
Absolute reflectance depends on grain size, packing, illumination angle and how
level the sample cup is - none of which say anything about mineralogy. Dividing
by the convex hull removes all of that and leaves band position, depth, width
and asymmetry, which are the properties that actually identify a mineral
(Clark & Roush 1984).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class ProcessedSpectrum:
    wavelength_nm: np.ndarray
    reflectance: np.ndarray          # resampled, smoothed reflectance
    continuum: np.ndarray            # fitted convex-hull continuum
    continuum_removed: np.ndarray    # reflectance / continuum, in [0, ~1]
    first_derivative: np.ndarray
    raw_counts: np.ndarray | None = None
    noise_sigma: float = 0.0     # noise sd of the continuum-removed spectrum
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "wavelength_nm": self.wavelength_nm.tolist(),
            "reflectance": np.round(self.reflectance, 6).tolist(),
            "continuum": np.round(self.continuum, 6).tolist(),
            "continuum_removed": np.round(self.continuum_removed, 6).tolist(),
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
#  Radiometry
# ---------------------------------------------------------------------------
def nonlinearity_correct(counts: np.ndarray, coeffs) -> np.ndarray:
    """
    Apply the USB4000's factory nonlinearity polynomial.

    Ocean Optics stores 8 coefficients in EEPROM; the corrected value is
    ``raw / polynomial(raw)``. Without this, deep absorption bands read a few
    percent too shallow, which biases band-depth-based identification.
    """
    if coeffs is None or len(coeffs) == 0:
        return counts
    c = np.asarray(coeffs, dtype=np.float64)
    poly = np.polyval(c[::-1], counts)
    poly = np.where(np.abs(poly) < 1e-9, 1.0, poly)
    return counts / poly


def to_reflectance(sample: np.ndarray, dark: np.ndarray | None,
                   white: np.ndarray | None, *,
                   floor: float = 1e-4) -> np.ndarray:
    """
    R = (S - D) / (W - D)

    Both the dark and the white reference must have been taken at the same
    integration time as the sample, or the ratio is meaningless. The device
    layer enforces that; this function only does the arithmetic.
    """
    s = np.asarray(sample, dtype=np.float64)
    if dark is None:
        dark = np.zeros_like(s)
    d = np.asarray(dark, dtype=np.float64)
    if white is None:
        # No white reference: return dark-corrected relative signal, normalised
        # to its own maximum. Usable for shape, not for absolute reflectance.
        rel = s - d
        peak = np.percentile(rel, 99.5)
        return np.clip(rel / max(peak, 1e-9), floor, 2.0)
    w = np.asarray(white, dtype=np.float64)
    denom = w - d
    denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
    r = (s - d) / denom
    if np.isnan(r).any():
        idx = np.arange(r.size)
        valid = np.isfinite(r)
        if valid.sum() >= 2:
            r = np.interp(idx, idx[valid], r[valid])
        else:
            r = np.full_like(r, floor)
    return np.clip(r, floor, 2.0)


# ---------------------------------------------------------------------------
#  Cleaning
# ---------------------------------------------------------------------------
def _robust_sigma(y: np.ndarray, window: int = 9) -> float:
    """Robust high-frequency noise sigma, from the residual after a local fit."""
    y = np.asarray(y, dtype=np.float64)
    if y.size < window + 2:
        return 0.0
    win = window if window % 2 == 1 else window + 1
    resid = y - savgol_filter(y, win, 3)
    return float(1.4826 * np.median(np.abs(resid - np.median(resid))))


def despike(y: np.ndarray, window: int = 9, z_thresh: float = 6.0) -> np.ndarray:
    """
    Remove cosmic-ray hits and hot pixels using a modified z-score on the
    residual from a running median. Deliberately conservative: real narrow
    mineral features (the apatite REE lines, for instance) are only a few
    channels wide and must survive.
    """
    y = np.asarray(y, dtype=np.float64).copy()
    n = y.size
    if n < window + 2:
        return y
    pad = window // 2
    padded = np.pad(y, pad, mode="edge")
    med = np.array([np.median(padded[i:i + window]) for i in range(n)])
    resid = y - med
    mad = np.median(np.abs(resid - np.median(resid)))
    if mad < 1e-12:
        return y
    z = 0.6745 * np.abs(resid) / mad
    spikes = z > z_thresh
    if spikes.any():
        y[spikes] = med[spikes]
    return y


def resample(wl_src: np.ndarray, y: np.ndarray, wl_dst: np.ndarray) -> np.ndarray:
    """
    Resample onto the analysis grid.

    The USB4000's native grid is a non-linear third-order polynomial in pixel
    index, so this is a genuine resample, not a relabelling.
    """
    wl_src = np.asarray(wl_src, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(wl_src)
    return np.interp(wl_dst, wl_src[order], y[order],
                     left=y[order][0], right=y[order][-1])


def smooth(y: np.ndarray, window_nm: float = 9.0, grid_step_nm: float = 1.0,
           polyorder: int = 3) -> np.ndarray:
    """
    Savitzky-Golay smoothing. A polynomial filter is used rather than a boxcar
    or Gaussian because it preserves peak height and position - a boxcar of the
    same width would shrink band depths and shift asymmetric bands.
    """
    win = int(round(window_nm / max(grid_step_nm, 1e-6)))
    if win % 2 == 0:
        win += 1
    win = max(win, polyorder + 2)
    if win >= y.size:
        win = y.size - 1 if (y.size - 1) % 2 == 1 else y.size - 2
    if win <= polyorder + 1:
        return np.asarray(y, dtype=np.float64)
    return savgol_filter(np.asarray(y, dtype=np.float64), win, polyorder)


def derivative(y: np.ndarray, wl: np.ndarray, order: int = 1,
               window_nm: float = 15.0) -> np.ndarray:
    """Savitzky-Golay derivative; sharpens band edges and cancels additive drift."""
    step = float(np.median(np.diff(wl)))
    win = int(round(window_nm / max(step, 1e-6)))
    if win % 2 == 0:
        win += 1
    win = max(win, 5)
    if win >= y.size:
        return np.gradient(y, wl, edge_order=1)
    return savgol_filter(y, win, 3, deriv=order, delta=step)


# ---------------------------------------------------------------------------
#  Continuum removal
# ---------------------------------------------------------------------------
def convex_hull_continuum(wl: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Upper convex hull of the spectrum, interpolated back onto the grid.

    Implemented with a monotone-chain scan rather than scipy's ConvexHull so the
    result is deterministic and the degenerate cases (flat or 2-point spectra)
    behave predictably.
    """
    wl = np.asarray(wl, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 3:
        return np.maximum(y, 1e-9)

    hull = []  # indices forming the upper hull
    for i in range(n):
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            cross = (wl[i2] - wl[i1]) * (y[i] - y[i1]) - (wl[i] - wl[i1]) * (y[i2] - y[i1])
            if cross >= 0:      # i2 lies on or below the line i1->i, drop it
                hull.pop()
            else:
                break
        hull.append(i)

    idx = np.array(hull, dtype=int)
    cont = np.interp(wl, wl[idx], y[idx])
    return np.maximum(cont, 1e-9)


def continuum_removal(wl: np.ndarray, y: np.ndarray):
    """Return (continuum, continuum-removed spectrum)."""
    cont = convex_hull_continuum(wl, y)
    cr = np.clip(np.asarray(y, dtype=np.float64) / cont, 0.0, 1.5)
    return cont, cr


# ---------------------------------------------------------------------------
#  Normalisations used by the learned model
# ---------------------------------------------------------------------------
def snv(y: np.ndarray) -> np.ndarray:
    """Standard normal variate: removes multiplicative scatter and offset."""
    y = np.asarray(y, dtype=np.float64)
    sd = y.std()
    return (y - y.mean()) / (sd if sd > 1e-12 else 1.0)


def vector_normalise(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n = np.linalg.norm(y)
    return y / (n if n > 1e-12 else 1.0)


# ---------------------------------------------------------------------------
#  Full pipeline
# ---------------------------------------------------------------------------
def process(wl_src: np.ndarray,
            sample_counts: np.ndarray,
            grid_nm: np.ndarray,
            *,
            dark: np.ndarray | None = None,
            white: np.ndarray | None = None,
            already_reflectance: bool = False,
            nonlinearity_coeffs=None,
            smooth_window_nm: float = 9.0) -> ProcessedSpectrum:
    """Run the full chain from raw detector counts to analysis-ready features."""
    raw = np.asarray(sample_counts, dtype=np.float64)

    if already_reflectance:
        refl_native = np.clip(raw, 1e-5, 2.0)
    else:
        s = nonlinearity_correct(raw, nonlinearity_coeffs)
        d = nonlinearity_correct(dark, nonlinearity_coeffs) if dark is not None else None
        w = nonlinearity_correct(white, nonlinearity_coeffs) if white is not None else None
        refl_native = to_reflectance(s, d, w)

    refl_native = despike(refl_native)
    refl_raw = resample(wl_src, refl_native, grid_nm)
    step = float(np.median(np.diff(grid_nm)))
    refl = smooth(refl_raw, smooth_window_nm, step)
    refl = np.clip(refl, 1e-5, 2.0)

    cont, cr = continuum_removal(grid_nm, refl)
    d1 = derivative(cr, grid_nm, order=1)

    # Noise level of the continuum-removed spectrum, needed downstream to
    # separate real absorption bands from artefacts. It has to be measured
    # BEFORE smoothing - afterwards the high-frequency residual is gone and any
    # estimate from it is far too optimistic - then propagated through the
    # smoother analytically. A Savitzky-Golay filter is linear, so it scales
    # white-noise sigma by the 2-norm of its coefficients, which is exact rather
    # than a rule of thumb.
    sigma_raw = _robust_sigma(refl_raw)
    win = int(round(smooth_window_nm / max(step, 1e-6)))
    win = win + 1 if win % 2 == 0 else win
    win = max(win, 5)
    try:
        from scipy.signal import savgol_coeffs
        gain = float(np.linalg.norm(savgol_coeffs(min(win, refl.size - 1 | 1), 3)))
    except Exception:
        gain = 1.0 / np.sqrt(max(win, 1))
    sigma_cr = sigma_raw * gain / max(float(np.median(cont)), 1e-9)

    return ProcessedSpectrum(
        wavelength_nm=np.asarray(grid_nm, dtype=np.float64),
        reflectance=refl,
        continuum=cont,
        continuum_removed=cr,
        first_derivative=d1,
        raw_counts=raw,
        noise_sigma=float(sigma_cr),
        meta={
            "noise_sigma_cr": round(float(sigma_cr), 6),
            "snr_estimate": round(float(np.median(refl) / max(sigma_raw, 1e-12)), 1),
            "already_reflectance": already_reflectance,
            "had_white_reference": white is not None,
            "had_dark_reference": dark is not None,
            "smooth_window_nm": smooth_window_nm,
        },
    )
