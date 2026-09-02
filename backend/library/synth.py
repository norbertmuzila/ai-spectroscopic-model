"""
Physical forward model for mineral reflectance spectra.

Two things live here:

* ``synthesize`` builds a reflectance spectrum for a mineral from its Modified
  Gaussian Model parameters. This gives the system a usable endmember for every
  mineral in the knowledge base even before the measured USGS library has been
  downloaded, and it is also how the training set is augmented over grain size,
  weathering and illumination.

* ``intimate_mix`` / ``areal_mix`` combine endmembers the way real samples do.
  Areal (checkerboard) mixing is linear in reflectance. Intimate mixing - grains
  in contact, which is what a powdered rock sample actually is - is strongly
  non-linear in reflectance but close to linear in single-scattering albedo, so
  it is done in Hapke w-space (Hapke 1981, 1993).
"""
from __future__ import annotations

import numpy as np

# Standard bidirectional viewing geometry: incidence 30 deg, emission 0 deg.
MU0 = float(np.cos(np.deg2rad(30.0)))
MU = 1.0


# ---------------------------------------------------------------------------
#  Hapke single-scattering albedo
# ---------------------------------------------------------------------------
def _h_function(x: np.ndarray | float, w: np.ndarray) -> np.ndarray:
    """Chandrasekhar H-function, Hapke's 2-stream approximation."""
    gamma = np.sqrt(np.clip(1.0 - w, 0.0, 1.0))
    return (1.0 + 2.0 * x) / (1.0 + 2.0 * x * gamma)


def ssa_to_reflectance(w: np.ndarray, mu0: float = MU0, mu: float = MU) -> np.ndarray:
    """Isotropic multiple-scattering approximation: w -> bidirectional reflectance."""
    w = np.clip(w, 0.0, 0.999999)
    return (w / (4.0 * (mu0 + mu))) * _h_function(mu0, w) * _h_function(mu, w)


def reflectance_to_ssa(r: np.ndarray, mu0: float = MU0, mu: float = MU,
                       iters: int = 60) -> np.ndarray:
    """Invert the IMSA relation numerically (monotonic, so bisection is safe)."""
    r = np.clip(np.asarray(r, dtype=np.float64), 1e-6, None)
    lo = np.zeros_like(r)
    hi = np.full_like(r, 0.999999)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_low = ssa_to_reflectance(mid, mu0, mu) < r
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
#  Modified Gaussian Model forward synthesis
# ---------------------------------------------------------------------------
def _mgm_component(wl_um: np.ndarray, centre: float, width: float,
                   strength: float) -> np.ndarray:
    """One modified Gaussian, Gaussian in energy (1/lambda) rather than wavelength."""
    inv = 1.0 / np.maximum(wl_um, 1e-9)
    inv_c = 1.0 / centre
    # ``width`` is given in micrometres of FWHM-equivalent; convert to the
    # corresponding sigma in inverse-micrometre space at the band centre.
    sigma_inv = width / (centre ** 2)
    return strength * np.exp(-((inv - inv_c) ** 2) / (2.0 * sigma_inv ** 2))


def _absorption_edge(wl_um: np.ndarray, lam0: float, width: float,
                     depth: float) -> np.ndarray:
    """
    Charge-transfer edge: strong absorption short of ``lam0``, transparent long
    of it. Modelled as a logistic in wavelength.
    """
    return depth / (1.0 + np.exp((wl_um - lam0) / max(width, 1e-6)))


def synthesize(mineral, wl_nm: np.ndarray, *,
               grain_size: float = 1.0,
               weathering: float = 0.0,
               band_scale: float = 1.0,
               shift_nm: float = 0.0,
               albedo_scale: float = 1.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Build a reflectance spectrum for ``mineral`` on the wavelength grid ``wl_nm``.

    Parameters
    ----------
    grain_size
        Relative grain size. Larger grains mean longer optical path, so deeper
        bands and lower continuum reflectance - the single biggest natural
        source of spectral variability in powdered samples.
    weathering
        0 to 1. Space weathering / coating: darkens, reddens and suppresses
        absorption bands, as nanophase iron does on the Moon and as dust
        coatings do on Mars.
    band_scale
        Extra multiplicative factor on band depths (mineral crystallinity,
        chemistry, packing).
    shift_nm
        Rigid wavelength shift of the whole feature set, standing in for
        solid-solution chemistry (e.g. Fe/Mg ratio moving pyroxene band centres).
    """
    wl_um = np.asarray(wl_nm, dtype=np.float64) / 1000.0
    wl_shift = wl_um - shift_nm / 1000.0
    wl_shift = np.maximum(wl_shift, 1e-6)

    path = float(np.clip(grain_size, 0.05, 20.0))
    depth_gain = band_scale * (path ** 0.45) * (1.0 - 0.85 * weathering)

    # Continuum in ln-reflectance: level + red slope, both affected by grain
    # size (bigger grains are darker) and weathering (darker and redder).
    level = mineral.albedo * albedo_scale
    level *= (path ** -0.22)
    level *= (1.0 - 0.72 * weathering)
    level = float(np.clip(level, 0.002, 0.98))

    slope = mineral.slope + 0.55 * weathering
    ln_r = np.log(level) + slope * (wl_um - 1.0)

    for centre, width, strength in mineral.bands:
        ln_r -= depth_gain * _mgm_component(wl_shift, centre, width, strength)

    if mineral.edge is not None:
        lam0, width, depth = mineral.edge
        edge_gain = (1.0 - 0.6 * weathering) * (path ** 0.25)
        ln_r -= edge_gain * _absorption_edge(wl_shift, lam0, width, depth)

    refl = np.exp(ln_r)

    if rng is not None:
        # Small correlated multiplicative texture, as real samples always show.
        texture = rng.normal(0.0, 0.006, size=refl.size)
        k = max(3, refl.size // 40)
        kernel = np.ones(k) / k
        texture = np.convolve(texture, kernel, mode="same")
        refl = refl * (1.0 + texture)

    return np.clip(refl, 1e-5, 1.0)


# ---------------------------------------------------------------------------
#  Mixing
# ---------------------------------------------------------------------------
def areal_mix(spectra: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Checkerboard / areal mixture: linear in reflectance."""
    f = np.asarray(fractions, dtype=np.float64)
    f = f / max(f.sum(), 1e-12)
    return np.tensordot(f, np.asarray(spectra, dtype=np.float64), axes=(0, 0))


def intimate_mix(spectra: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """
    Intimate (particulate) mixture. Linear in single-scattering albedo, which is
    what a ground-up rock sample actually produces. Dark endmembers dominate far
    more than their mass fraction would suggest - this non-linearity is exactly
    why linear unmixing alone underestimates opaque phases such as magnetite.
    """
    f = np.asarray(fractions, dtype=np.float64)
    f = f / max(f.sum(), 1e-12)
    w = reflectance_to_ssa(np.asarray(spectra, dtype=np.float64))
    w_mix = np.tensordot(f, w, axes=(0, 0))
    return ssa_to_reflectance(w_mix)


def mix(spectra: np.ndarray, fractions: np.ndarray, mode: str = "intimate") -> np.ndarray:
    return intimate_mix(spectra, fractions) if mode == "intimate" else areal_mix(spectra, fractions)
