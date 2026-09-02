"""
Absorption-feature extraction.

Everything here operates on the continuum-removed spectrum, where an absorption
band is a well at some wavelength. For each band we measure the four quantities
that mineralogists actually use to identify a phase:

    centre      - where the minimum is, refined to sub-channel precision by a
                  parabolic fit. Band centre is chemistry: pyroxene Band I moves
                  from 0.90 to 1.05 um as Ca and Fe increase.
    depth       - 1 - R_continuum_removed at the minimum. Depth is abundance and
                  grain size, not identity, which is why it is a weak feature on
                  its own and a strong one in combination.
    width       - full width at half the band depth. Distinguishes a broad
                  electronic transition from a narrow vibrational overtone.
    asymmetry   - the olivine 1.05 um composite is strongly asymmetric because
                  it is three overlapping bands; a single pyroxene band is not.

These are handed to the gradient-boosted model as explicit features. The CNN
sees the raw continuum-removed vector instead, so the two model families fail
in different ways - which is the point of ensembling them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import find_peaks

# Bump whenever band detection or the feature vector changes in a way that makes
# previously-extracted features incompatible. The training-set cache and the
# saved model both record it, so a stale cache or a model trained against older
# feature semantics is detected rather than silently reused - which would leave
# the classifier scoring features that no longer mean what it learned.
FEATURE_VERSION = 2

# NumPy 2 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class AbsorptionBand:
    centre_nm: float
    depth: float
    width_nm: float
    asymmetry: float
    area: float
    prominence: float

    def as_dict(self) -> dict:
        return {k: round(float(v), 5) for k, v in asdict(self).items()}


def _parabolic_refine(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """Sub-channel minimum position via a 3-point parabolic fit."""
    if i <= 0 or i >= x.size - 1:
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(x[i])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    step = float(x[i + 1] - x[i - 1]) / 2.0
    return float(x[i] + delta * step)


# Noise must be measured at the scale band detection works at. A 9-point
# estimate only sees channel-to-channel scatter, which Savitzky-Golay smoothing
# has already removed; the structure that actually creates false bands is the
# residual wiggle on a 20-30 nm scale that survives smoothing. Real absorption
# bands are broader than this, so they still register as signal.
NOISE_WINDOW = 25


def estimate_noise(y: np.ndarray, window: int = NOISE_WINDOW) -> float:
    """
    Robust noise level at the band-detection scale, as a standard deviation.

    Taken from the residual after a local cubic fit over ``window`` channels and
    scaled by the MAD constant. Anything narrower than the window counts as
    noise; anything broader is treated as real structure.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size < window + 2:
        return 0.0
    from scipy.signal import savgol_filter
    win = window if window % 2 == 1 else window + 1
    resid = y - savgol_filter(y, win, 3)
    return float(1.4826 * np.median(np.abs(resid - np.median(resid))))


def find_bands(wl_nm: np.ndarray, continuum_removed: np.ndarray, *,
               min_depth: float = 0.012,
               max_bands: int = 12,
               noise_sigma: float | None = None,
               n_sigma: float = 4.0,
               max_width_frac: float = 0.55) -> list:
    """
    Detect absorption bands in a continuum-removed spectrum.

    When ``noise_sigma`` is supplied the detection threshold becomes
    ``max(min_depth, n_sigma * noise_sigma)``. This matters more than it looks.
    Convex-hull continuum removal applied to a noisy, featureless spectrum
    manufactures apparent absorption bands several percent deep out of nothing,
    and a dark sample - magnetite, ilmenite, a shadowed surface - is exactly
    where reflectance is a ratio of two small numbers and therefore noisiest.
    A fixed threshold lets those artefacts drive a confident identification of a
    mineral that has real bands where the noise happened to dip. Requiring a
    band to stand 4 sigma above the measurement's own noise floor is the
    difference between reporting evidence and reporting noise.
    """
    wl = np.asarray(wl_nm, dtype=np.float64)
    cr = np.asarray(continuum_removed, dtype=np.float64)
    inverted = 1.0 - cr

    if noise_sigma is None:
        noise_sigma = estimate_noise(cr)
    min_depth = max(min_depth, n_sigma * float(noise_sigma))

    step = float(np.median(np.diff(wl)))
    min_sep = max(3, int(round(12.0 / max(step, 1e-6))))

    peaks, props = find_peaks(inverted, height=min_depth, distance=min_sep,
                              prominence=min_depth * 0.5)
    if peaks.size == 0:
        return []

    order = np.argsort(-props["prominences"])
    peaks = peaks[order][:max_bands]
    proms = props["prominences"][order][:max_bands]

    bands = []
    for pk, prom in zip(peaks, proms):
        depth = float(inverted[pk])
        centre = _parabolic_refine(wl, cr, int(pk))

        half = depth / 2.0
        left = int(pk)
        while left > 0 and inverted[left] > half:
            left -= 1
        right = int(pk)
        while right < inverted.size - 1 and inverted[right] > half:
            right += 1
        width = float(wl[right] - wl[left])

        # Reject anything too wide to be a band. Convex-hull continuum removal
        # applied to a gently curved spectrum - a dark opaque, or a mineral whose
        # real absorption lies just outside the measured range - manufactures a
        # broad shallow "band" spanning most of the window. It has a depth, so a
        # depth threshold alone passes it, and it then matches any reference with
        # a similarly bowed continuum. But its centre is not localised, so it
        # carries no diagnostic information: a feature that wide is continuum
        # curvature, not an absorption band, and treating it as one is how a
        # featureless magnetite spectrum turns into a confident chromite call.
        if width > max_width_frac * float(wl[-1] - wl[0]):
            continue

        left_area = float(_trapz(inverted[left:pk + 1], wl[left:pk + 1])) if pk > left else 0.0
        right_area = float(_trapz(inverted[pk:right + 1], wl[pk:right + 1])) if right > pk else 0.0
        total = left_area + right_area
        asym = (left_area - right_area) / total if total > 1e-12 else 0.0

        bands.append(AbsorptionBand(
            centre_nm=centre, depth=depth, width_nm=width,
            asymmetry=float(asym), area=total, prominence=float(prom),
        ))

    bands.sort(key=lambda b: b.centre_nm)
    return bands


# ---------------------------------------------------------------------------
#  Fixed-length feature vector for the tabular model
# ---------------------------------------------------------------------------
#  Windows chosen to straddle the diagnostic regions the VNIR range can reach.
BAND_WINDOWS_NM = [
    (400, 460), (460, 520), (520, 580), (580, 640), (640, 700),
    (700, 780), (780, 840), (840, 900), (900, 960), (960, 1020),
    (1020, 1100), (1100, 1300), (1350, 1450), (1850, 1980),
    (2100, 2200), (2200, 2280), (2280, 2360), (2450, 2560),
]


def window_stats(wl_nm: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """
    Per-window depth and centre-of-mass, for every window that the instrument's
    wavelength range actually covers. Windows outside the range yield zeros and
    a coverage flag of 0, so the model can learn "this evidence was unavailable"
    rather than "this evidence was absent" - a distinction that matters enormously
    when a VNIR-only instrument looks at a SWIR-diagnostic mineral.
    """
    wl = np.asarray(wl_nm, dtype=np.float64)
    cr = np.asarray(cr, dtype=np.float64)
    feats = []
    for lo, hi in BAND_WINDOWS_NM:
        m = (wl >= lo) & (wl <= hi)
        if m.sum() < 3:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        seg_wl, seg = wl[m], cr[m]
        depth = float(1.0 - seg.min())
        pos = float(seg_wl[int(np.argmin(seg))])
        pos_norm = (pos - lo) / max(hi - lo, 1e-9)
        area = float(_trapz(1.0 - seg, seg_wl) / max(hi - lo, 1e-9))
        feats.extend([depth, pos_norm, area, 1.0])
    return np.asarray(feats, dtype=np.float64)


def slope_features(wl_nm: np.ndarray, reflectance: np.ndarray) -> np.ndarray:
    """
    Continuum-slope descriptors. These carry the information continuum removal
    deliberately discards, and it is real information: the red slope is how you
    tell space-weathered lunar regolith or Martian dust from fresh rock.
    """
    wl = np.asarray(wl_nm, dtype=np.float64)
    r = np.clip(np.asarray(reflectance, dtype=np.float64), 1e-5, None)

    def band_mean(lo, hi):
        m = (wl >= lo) & (wl <= hi)
        return float(r[m].mean()) if m.sum() else np.nan

    v = band_mean(450, 550)
    g = band_mean(550, 650)
    red = band_mean(650, 750)
    nir = band_mean(850, 950)
    nir2 = band_mean(950, 1050)

    def ratio(a, b):
        if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
            return 0.0
        return float(np.clip(a / b, 0.0, 20.0))

    ln_r = np.log(r)
    ok = np.isfinite(ln_r)
    slope = float(np.polyfit(wl[ok] / 1000.0, ln_r[ok], 1)[0]) if ok.sum() > 3 else 0.0

    return np.asarray([
        ratio(red, v),        # visible redness - ferric / space weathering
        ratio(nir, red),
        ratio(nir2, nir),     # 1 um band shoulder
        ratio(g, v),
        slope,                # overall ln-reflectance slope per micrometre
        float(np.nanmean(r)),
        float(np.nanmax(r) - np.nanmin(r)),
    ], dtype=np.float64)


def feature_vector(wl_nm: np.ndarray, reflectance: np.ndarray,
                   continuum_removed: np.ndarray) -> np.ndarray:
    """Full tabular feature vector used by the gradient-boosted classifier."""
    bands = find_bands(wl_nm, continuum_removed)
    top = sorted(bands, key=lambda b: -b.depth)[:4]
    band_feats = []
    for i in range(4):
        if i < len(top):
            b = top[i]
            band_feats.extend([b.centre_nm / 1000.0, b.depth,
                               b.width_nm / 1000.0, b.asymmetry])
        else:
            band_feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.concatenate([
        window_stats(wl_nm, continuum_removed),
        slope_features(wl_nm, reflectance),
        np.asarray(band_feats, dtype=np.float64),
        [float(len(bands))],
    ])


FEATURE_DIM = len(BAND_WINDOWS_NM) * 4 + 7 + 16 + 1
