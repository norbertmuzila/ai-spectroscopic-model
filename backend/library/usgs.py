"""
Reader for the USGS Spectral Library (splib07a, ASCII distribution).

The ASCII release is laid out as::

    ASCIIdata_splib07a/
        splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt
        splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt
        ...
        ChapterM_Minerals/
            splib07a_Hematite_GDS27_...__BECKa_AREF.txt
        ChapterS_SoilsAndMixtures/
        ...

Every spectrum file is a single title line followed by one reflectance value per
line, on the wavelength grid named by the instrument code embedded in the file
name (BECK, ASDFR, AVIRIS, NIC4 and so on). Deleted or invalid channels are
flagged with the sentinel -1.23e34 and are masked out here.

Run ``scripts/fetch_usgs_library.py`` to obtain the data; this module only
parses whatever is present on disk, and the rest of the system degrades
gracefully to synthetic endmembers when nothing is.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

DELETED = -1.23e34
DELETED_TOL = 1e33


def _read_values(path: Path) -> np.ndarray:
    """Read a splib ASCII file: one header line, then one float per line."""
    values = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        fh.readline()  # title line
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue
    return np.asarray(values, dtype=np.float64)


def _instrument_token(filename: str) -> str:
    """
    Pull the instrument code out of a spectrum file name.

    ``splib07a_Hematite_GDS27_..._BECKa_AREF.txt`` -> ``BECK``
    Trailing lowercase revision letters are stripped.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    for part in reversed(parts):
        if part.upper() in {"AREF", "RREF", "TRAN", "ASDFRA"}:
            continue
        token = re.sub(r"[a-z0-9]+$", "", part)
        if token and token.isupper() and len(token) >= 3:
            return token
    return ""


def _mineral_field(filename: str) -> str:
    """``splib07a_Hematite_GDS27_..._AREF.txt`` -> ``hematite gds27``."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if parts and parts[0].lower().startswith("splib"):
        parts = parts[1:]
    keep = []
    for part in parts:
        if part.upper() in {"AREF", "RREF", "TRAN"}:
            break
        keep.append(part)
    return " ".join(keep).lower()


def load_wavelength_grids(root: Path) -> dict:
    """Map instrument code -> wavelength array in nanometres."""
    grids: dict = {}
    for path in root.rglob("*Wavelengths*.txt"):
        stem = path.stem
        m = re.search(r"Wavelengths_([A-Za-z0-9]+)", stem)
        if not m:
            continue
        code = m.group(1).upper()
        micron = _read_values(path)
        if micron.size == 0:
            continue
        micron = np.where(np.abs(micron - DELETED) < DELETED_TOL, np.nan, micron)
        grids[code] = micron * 1000.0  # micrometres -> nanometres
    return grids


def _match_grid(grids: dict, token: str):
    if not grids:
        return None
    if token in grids:
        return grids[token]
    for code, grid in grids.items():
        if token.startswith(code) or code.startswith(token):
            return grid
    return None


def load_spectra(root: Path, minerals, target_grid_nm: np.ndarray,
                 max_per_mineral: int = 12) -> list:
    """
    Parse every splib spectrum under ``root`` and return the ones that match a
    mineral in the knowledge base, resampled onto ``target_grid_nm``.

    Returns a list of dicts: name, sample_id, reflectance, coverage.
    """
    root = Path(root)
    if not root.exists():
        return []

    grids = load_wavelength_grids(root)
    if not grids:
        return []

    # Build hint -> mineral lookup, longest hints first so "nanophase hematite"
    # is preferred over the substring "hematite".
    hints = [(m.usgs_hint.lower(), m) for m in minerals if m.usgs_hint]
    hints.sort(key=lambda kv: -len(kv[0]))

    found: dict = {}
    out = []

    for path in sorted(root.rglob("splib07*.txt")):
        if "Wavelength" in path.name:
            continue
        field = _mineral_field(path.name)
        mineral = None
        for hint, cand in hints:
            if hint in field:
                mineral = cand
                break
        if mineral is None:
            continue
        if found.get(mineral.name, 0) >= max_per_mineral:
            continue

        token = _instrument_token(path.name)
        grid = _match_grid(grids, token)
        if grid is None:
            continue

        refl = _read_values(path)
        if refl.size != grid.size:
            continue

        good = (np.abs(refl - DELETED) > DELETED_TOL) & np.isfinite(refl) \
            & np.isfinite(grid) & (refl > -0.05)
        if good.sum() < 20:
            continue

        wl_good = grid[good]
        r_good = np.clip(refl[good], 1e-5, 1.5)
        order = np.argsort(wl_good)
        wl_good, r_good = wl_good[order], r_good[order]

        in_range = (target_grid_nm >= wl_good[0]) & (target_grid_nm <= wl_good[-1])
        coverage = float(in_range.mean())
        if coverage < 0.60:
            continue

        resampled = np.interp(target_grid_nm, wl_good, r_good,
                              left=np.nan, right=np.nan)
        # Edge-fill any small out-of-range tail so the vector is complete.
        if np.isnan(resampled).any():
            idx = np.arange(resampled.size)
            valid = ~np.isnan(resampled)
            resampled = np.interp(idx, idx[valid], resampled[valid])

        found[mineral.name] = found.get(mineral.name, 0) + 1
        out.append({
            "name": mineral.name,
            "sample_id": path.stem,
            "reflectance": resampled.astype(np.float32),
            "coverage": coverage,
            "instrument": token,
        })

    return out
