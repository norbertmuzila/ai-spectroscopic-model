"""
The USGS splib05a library as the system's ground truth.

Everything the engine reports about a mineral is derived here, from the measured
library, rather than from a taxonomy written by hand:

* **Names and groups** come from each spectrum's own title line, so a result
  names the same mineral USGS names, and carries the sample number (GDS27,
  HS22.3B, NMNH133746) that identifies exactly which library entry it matched.
  There is no separate vocabulary that could disagree with the source.

* **Diagnostic band positions are measured, not asserted.** An earlier version
  carried hand-entered band centres from the literature. Those are reasonable
  and they are also, unavoidably, approximate - and every downstream decision
  about what this instrument can and cannot see depended on them. Detecting the
  bands in the USGS spectrum itself replaces an approximation with the thing it
  was approximating.

* **The full 0.2-3.0 um range is retained** even though the USB4000 sees only
  340-912 nm, because knowing that a mineral's diagnostic band sits at 2.2 um is
  precisely what lets the engine say "this instrument cannot decide that".

Reference spectra are averaged per mineral only when they genuinely agree;
where a mineral's samples differ they are kept as separate references, because
two samples of the same mineral really can look different and collapsing them
into a mean produces a curve that matches neither.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend import config as cfg_mod

SPLIB_PATH = cfg_mod.ROOT / "data" / "library" / "splib05a" / "splib05a.json"

# Reflectance types worth using. AREF is absolute reflectance, which is what a
# dark/white-referenced bench measurement produces. Transmission spectra measure
# a different physical quantity and must not be mixed in.
USABLE_TYPES = {"AREF", "RREF", "REF", ""}


@dataclass
class UsgsSpectrum:
    name: str                  # USGS mineral name, e.g. "Hematite"
    sample: str                # USGS sample number, e.g. "GDS27"
    group: str                 # USGS mineral group, may be empty
    chapter: str
    title: str
    wavelength_nm: np.ndarray  # native grid, full range
    reflectance: np.ndarray
    file: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} {self.sample}".strip()

    def resample(self, grid_nm: np.ndarray) -> np.ndarray:
        """Interpolate onto an analysis grid, NaN outside the measured range."""
        wl, r = self.wavelength_nm, self.reflectance
        out = np.interp(grid_nm, wl, r, left=np.nan, right=np.nan)
        return out

    def covers(self, lo_nm: float, hi_nm: float, min_fraction: float = 0.9) -> bool:
        wl = self.wavelength_nm
        want = np.linspace(lo_nm, hi_nm, 64)
        inside = (want >= wl.min()) & (want <= wl.max())
        return float(inside.mean()) >= min_fraction


@dataclass
class SplibIndex:
    spectra: list = field(default_factory=list)
    citation: str = ""
    url: str = ""

    def names(self) -> list:
        return sorted({s.name for s in self.spectra})

    def by_name(self, name: str) -> list:
        return [s for s in self.spectra if s.name == name]

    def summary(self) -> dict:
        from collections import Counter
        per = Counter(s.name for s in self.spectra)
        return {
            "n_spectra": len(self.spectra),
            "n_minerals": len(per),
            "minerals_with_multiple_samples": sum(1 for v in per.values() if v > 1),
            "citation": self.citation,
        }


def available() -> bool:
    return SPLIB_PATH.exists()


_cache: SplibIndex | None = None


def load(reload: bool = False) -> SplibIndex:
    """Load the downloaded splib05a JSON. Empty index if it has not been fetched."""
    global _cache
    if _cache is not None and not reload:
        return _cache
    if not SPLIB_PATH.exists():
        _cache = SplibIndex()
        return _cache

    blob = json.loads(SPLIB_PATH.read_text(encoding="utf-8"))
    out = []
    for rec in blob.get("spectra", []):
        if rec.get("type", "").upper() not in USABLE_TYPES:
            continue
        wl = np.asarray(rec["wavelength_nm"], dtype=np.float64)
        r = np.asarray(rec["reflectance"], dtype=np.float64)
        good = np.isfinite(wl) & np.isfinite(r) & (r > -0.05) & (r < 2.0)
        if good.sum() < 30:
            continue
        wl, r = wl[good], r[good]
        order = np.argsort(wl)
        out.append(UsgsSpectrum(
            name=rec.get("name", "Unknown").strip(),
            sample=rec.get("sample", "").strip(),
            group=rec.get("group", "").strip(),
            chapter=rec.get("chapter", ""),
            title=rec.get("title", ""),
            file=rec.get("file", ""),
            wavelength_nm=wl[order],
            reflectance=np.clip(r[order], 1e-5, 1.5),
        ))

    _cache = SplibIndex(spectra=out,
                        citation=blob.get("citation", ""),
                        url=blob.get("url", ""))
    return _cache


# ---------------------------------------------------------------------------
#  Diagnostic bands, measured from the library rather than asserted
# ---------------------------------------------------------------------------
def measure_diagnostic_bands(spec: UsgsSpectrum,
                             lo_nm: float = 350.0,
                             hi_nm: float = 2600.0,
                             min_depth: float = 0.03,
                             max_bands: int = 8) -> list:
    """
    Find a mineral's diagnostic absorption bands in its own USGS spectrum.

    Run over the full library range, not the instrument's range - the whole
    point is to know which of a mineral's identifying features fall outside what
    the USB4000 can see, and a band that was never looked for cannot be reported
    as missing.
    """
    from backend.processing import features as featmod
    from backend.processing import preprocess as prep

    wl, r = spec.wavelength_nm, spec.reflectance
    m = (wl >= lo_nm) & (wl <= hi_nm)
    if m.sum() < 40:
        return []
    w, y = wl[m], r[m]

    # Regularise onto an even grid: the library's native sampling is uneven, and
    # convex-hull continuum removal on an uneven grid biases band widths.
    grid = np.arange(w.min(), w.max(), 2.0)
    yy = np.interp(grid, w, y)
    _, cr = prep.continuum_removal(grid, yy)
    bands = featmod.find_bands(grid, cr, min_depth=min_depth, max_bands=max_bands)
    return [{"centre_nm": round(b.centre_nm, 1),
             "depth": round(b.depth, 4),
             "width_nm": round(b.width_nm, 1)} for b in bands]


# ---------------------------------------------------------------------------
#  Cached per-mineral diagnostic bands
# ---------------------------------------------------------------------------
BANDS_PATH = cfg_mod.ROOT / "data" / "library" / "splib05a" / "diagnostic_bands.json"

_bands_cache: dict | None = None


def diagnostic_bands(rebuild: bool = False) -> dict:
    """
    Diagnostic band centres for every USGS mineral, measured over the full
    0.2-3.0 um library range and cached.

    This is what lets the engine say a mineral's identifying features lie
    outside the instrument's window. It has to be computed over the whole
    library range, not the instrument's: a band nobody looked for cannot be
    reported as out of reach, and the coverage check would then silently pass
    everything - which is exactly what happens when this is missing.
    """
    global _bands_cache
    if _bands_cache is not None and not rebuild:
        return _bands_cache

    if BANDS_PATH.exists() and not rebuild:
        try:
            _bands_cache = json.loads(BANDS_PATH.read_text(encoding="utf-8"))
            return _bands_cache
        except Exception:
            pass

    index = load()
    out: dict = {}
    for spec in index.spectra:
        bands = measure_diagnostic_bands(spec)
        prev = out.get(spec.name)
        # Keep the sample showing the clearest features: a mineral is
        # identifiable if any of its samples shows the band, and a weathered or
        # fine-grained sample would otherwise erase evidence the mineral has.
        if prev is None or sum(b["depth"] for b in bands) > sum(b["depth"] for b in prev):
            out[spec.name] = bands

    BANDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BANDS_PATH.write_text(json.dumps(out), encoding="utf-8")
    _bands_cache = out
    return out
