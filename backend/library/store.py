"""
The unified spectral library.

Combines measured USGS splib07a spectra (when present on disk) with physically
synthesised MGM endmembers for every mineral in the knowledge base, on one
common wavelength grid. Measured spectra always take precedence; synthetic ones
fill the gaps so the system is never blind to a mineral it knows about.

Every entry records its provenance, and that provenance is carried all the way
through to the analysis report - an identification supported by a measured USGS
reference is a materially stronger claim than one supported only by a modelled
endmember, and the report says which it was.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend import config as cfg_mod
from backend.library import minerals as mindb
from backend.library import synth, usgs


@dataclass
class SpectralLibrary:
    grid_nm: np.ndarray            # (W,) common wavelength grid
    spectra: np.ndarray            # (N, W) reflectance
    names: list                    # (N,) mineral name per entry
    sample_ids: list               # (N,) provenance id
    sources: list                  # (N,) 'usgs' | 'synthetic'
    classes: list                  # unique mineral names, ordered
    class_index: np.ndarray        # (N,) index into classes

    # ---- lookups ---------------------------------------------------------
    def mineral(self, name: str):
        return mindb.BY_NAME.get(name)

    def entries_for(self, name: str) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.names) == name)

    def class_spectra(self) -> np.ndarray:
        """One representative (median) spectrum per class, shape (C, W)."""
        out = np.zeros((len(self.classes), self.grid_nm.size), dtype=np.float64)
        for i, name in enumerate(self.classes):
            idx = self.entries_for(name)
            out[i] = np.median(self.spectra[idx], axis=0)
        return out

    def source_counts(self) -> dict:
        return {
            "usgs": int(sum(1 for s in self.sources if s == "usgs")),
            "synthetic": int(sum(1 for s in self.sources if s == "synthetic")),
        }

    def summary(self) -> dict:
        measured = {n for n, s in zip(self.names, self.sources) if s == "usgs"}
        return {
            "n_entries": int(self.spectra.shape[0]),
            "n_classes": len(self.classes),
            "grid_nm": [float(self.grid_nm[0]), float(self.grid_nm[-1])],
            "grid_points": int(self.grid_nm.size),
            "sources": self.source_counts(),
            "classes_with_measured_reference": sorted(measured),
        }

    # ---- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            grid_nm=self.grid_nm,
            spectra=self.spectra.astype(np.float32),
            names=np.array(self.names, dtype=object),
            sample_ids=np.array(self.sample_ids, dtype=object),
            sources=np.array(self.sources, dtype=object),
            classes=np.array(self.classes, dtype=object),
            class_index=self.class_index,
        )

    @classmethod
    def load(cls, path: Path) -> "SpectralLibrary":
        data = np.load(Path(path), allow_pickle=True)
        return cls(
            grid_nm=data["grid_nm"],
            spectra=data["spectra"].astype(np.float64),
            names=list(data["names"]),
            sample_ids=list(data["sample_ids"]),
            sources=list(data["sources"]),
            classes=list(data["classes"]),
            class_index=data["class_index"],
        )


# ---------------------------------------------------------------------------
#  Build
# ---------------------------------------------------------------------------
def build(grid_nm: np.ndarray | None = None,
          environments: list | None = None,
          usgs_dir: Path | None = None,
          allow_synthetic: bool = True,
          synthetic_variants: int = 5,
          seed: int = 7) -> SpectralLibrary:
    cfg = cfg_mod.load_config()
    if grid_nm is None:
        grid_nm = cfg_mod.analysis_grid()
    if environments is None:
        environments = []
        if cfg.get_path("library.include_terrestrial", True):
            environments.append(mindb.TERRESTRIAL)
        if cfg.get_path("library.include_martian", True):
            environments.append(mindb.MARS)
        if cfg.get_path("library.include_lunar", True):
            environments.append(mindb.LUNAR)
    if usgs_dir is None:
        usgs_dir = cfg.resolve("library.usgs_dir")

    selected = mindb.select(environments)
    rng = np.random.default_rng(seed)

    names, sample_ids, sources, rows = [], [], [], []

    # 1. Measured USGS references.
    measured = usgs.load_spectra(Path(usgs_dir), selected, grid_nm)
    for rec in measured:
        names.append(rec["name"])
        sample_ids.append(rec["sample_id"])
        sources.append("usgs")
        rows.append(rec["reflectance"].astype(np.float64))
    measured_names = set(names)

    # 2. Synthetic endmembers. A small spread of grain size and weathering is
    #    generated for every mineral so the matcher sees the natural range of
    #    band depth and continuum slope, not a single idealised curve.
    if allow_synthetic:
        for m in selected:
            n_variants = 1 if m.name in measured_names else synthetic_variants
            for v in range(n_variants):
                if n_variants == 1:
                    grain, weath, bscale = 1.0, 0.0, 1.0
                else:
                    grain = float(np.exp(rng.uniform(np.log(0.25), np.log(4.0))))
                    weath = float(rng.uniform(0.0, 0.45))
                    bscale = float(rng.uniform(0.75, 1.25))
                spec = synth.synthesize(m, grid_nm, grain_size=grain,
                                        weathering=weath, band_scale=bscale)
                names.append(m.name)
                sample_ids.append(f"SYN_{m.name.replace(' ', '')}_{v:02d}")
                sources.append("synthetic")
                rows.append(spec)

    spectra = np.vstack(rows) if rows else np.zeros((0, grid_nm.size))
    classes = sorted(set(names))
    cindex = {c: i for i, c in enumerate(classes)}
    class_index = np.array([cindex[n] for n in names], dtype=np.int32)

    return SpectralLibrary(
        grid_nm=np.asarray(grid_nm, dtype=np.float64),
        spectra=spectra,
        names=names,
        sample_ids=sample_ids,
        sources=sources,
        classes=classes,
        class_index=class_index,
    )


_cached: SpectralLibrary | None = None


def get_library(rebuild: bool = False) -> SpectralLibrary:
    """Load the cached library from disk, building and caching it if needed."""
    global _cached
    if _cached is not None and not rebuild:
        return _cached

    cfg = cfg_mod.load_config()
    path = cfg.resolve("library.store_path")
    grid = cfg_mod.analysis_grid()

    if path.exists() and not rebuild:
        lib = SpectralLibrary.load(path)
        if lib.grid_nm.size == grid.size and np.allclose(lib.grid_nm, grid):
            _cached = lib
            return lib

    lib = build(grid_nm=grid,
                allow_synthetic=bool(cfg.get_path("library.allow_synthetic_fallback", True)))
    lib.save(path)
    _cached = lib
    return lib
