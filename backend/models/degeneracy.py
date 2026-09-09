"""
Spectral degeneracy groups.

Over a restricted wavelength range, many minerals are not merely *hard* to tell
apart - they are genuinely indistinguishable, because the features that separate
them lie outside the range and never reached the detector. Below 1000 nm quartz,
calcite, gypsum, halite, kaolinite, muscovite and a couple of dozen others are
all the same measurement: a bright, featureless curve.

Reporting a 29-member prediction set for such a sample is technically correct
and practically useless. Reporting

    "Spectrally featureless, high albedo - one of 24 phases including quartz,
     calcite, gypsum, halite, kaolinite. SWIR coverage (1.4-2.5 um) is required
     to separate these."

says the same thing in a form a person can act on. That is what this module
builds: equivalence classes of minerals that this instrument cannot separate,
derived from the library itself rather than hand-written, so they update
automatically when the wavelength range or the library changes.

Distance between two minerals combines:

* the spectral angle between their continuum-removed absorption profiles, which
  is what identifies a mineral, and
* their albedo difference in log space, because a valid white reference makes
  magnetite (4% reflectance) and halite (88%) trivially separable even though
  both are featureless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.processing import features as featmod
from backend.processing import preprocess as prep

# A mineral whose deepest band is shallower than this has no usable absorption
# feature in range and is classified as featureless.
FEATURELESS_DEPTH = 0.025
# Cluster cut-off, in the combined distance units defined below (degrees).
DEFAULT_THRESHOLD = 9.0
# One decade of albedo difference counts as this many degrees of spectral angle.
ALBEDO_WEIGHT = 26.0


@dataclass
class DegeneracyMap:
    range_nm: tuple
    labels: dict          # mineral -> group id
    members: dict         # group id -> [minerals]
    names: dict           # group id -> human-readable label
    depths: dict          # mineral -> deepest band depth in range
    albedos: dict         # mineral -> mean reflectance in range

    def group_of(self, mineral: str) -> int:
        return self.labels.get(mineral, -1)

    def siblings(self, mineral: str) -> list:
        gid = self.labels.get(mineral)
        return [m for m in self.members.get(gid, []) if m != mineral] if gid is not None else []

    def label_for(self, mineral: str) -> str:
        gid = self.labels.get(mineral)
        return self.names.get(gid, mineral) if gid is not None else mineral

    def collapse(self, minerals: list) -> list:
        """Collapse a list of minerals into (group_label, members) pairs."""
        seen, out = set(), []
        for m in minerals:
            gid = self.labels.get(m, -1)
            if gid in seen:
                continue
            seen.add(gid)
            group_members = [x for x in self.members.get(gid, [m]) if x in minerals]
            out.append({
                "label": self.names.get(gid, m),
                "members": group_members,
                "n_in_group": len(self.members.get(gid, [m])),
            })
        return out

    def summary(self) -> dict:
        sizes = [len(v) for v in self.members.values()]
        return {
            "range_nm": [round(v, 1) for v in self.range_nm],
            "n_groups": len(self.members),
            "n_singleton_groups": int(sum(1 for s in sizes if s == 1)),
            "largest_group": int(max(sizes)) if sizes else 0,
        }


def _group_name(members: list, depths: dict, albedos: dict, mineral_lookup) -> str:
    """Human-readable label for an equivalence class."""
    if len(members) == 1:
        return members[0]

    mean_depth = float(np.mean([depths.get(m, 0.0) for m in members]))
    mean_alb = float(np.mean([albedos.get(m, 0.3) for m in members]))

    if mean_depth < FEATURELESS_DEPTH:
        tone = ("very dark" if mean_alb < 0.10 else
                "dark" if mean_alb < 0.25 else
                "moderate-albedo" if mean_alb < 0.55 else "bright")
        return f"Spectrally featureless, {tone}"

    # Named after the mineral groups actually present, which is the level at
    # which the answer is still true. The group comes from USGS's own
    # classification wherever the library provides one.
    groups = []
    for m in members:
        g = mineral_lookup(m)
        if isinstance(g, str):
            g = g.strip()
        elif g is not None:
            g = getattr(g, "group", None)
        if g and g not in groups:
            groups.append(g)
    if len(groups) == 1:
        return f"{groups[0]} group"
    if len(groups) <= 3:
        return " / ".join(groups)
    return f"{members[0]} and {len(members) - 1} spectrally similar phases"


def build(library, lo_nm: float, hi_nm: float,
          threshold: float = DEFAULT_THRESHOLD) -> DegeneracyMap:
    """Cluster the library's minerals by what this wavelength range can separate."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    grid = library.grid_nm
    mask = (grid >= lo_nm - 1e-6) & (grid <= hi_nm + 1e-6)
    if mask.sum() < 10:
        mask = np.ones_like(grid, dtype=bool)
    wl = grid[mask]

    classes = list(library.classes)
    cols = np.flatnonzero(mask)
    profiles, depths, albedos = [], {}, {}
    for name in classes:
        idx = library.entries_for(name)
        # Use the library entry with the *deepest* bands, not the median. The
        # question this map answers is "can this instrument separate these two
        # minerals in principle", so each mineral should be represented by its
        # best-case detectable spectrum. A median over weathered and
        # fine-grained variants washes out bands and would wrongly declare a
        # perfectly identifiable mineral featureless.
        best_profile, best_depth, best_spec = None, -1.0, None
        for row in idx:
            spec = library.spectra[row, cols]
            _, cr = prep.continuum_removal(wl, spec)
            absorption = 1.0 - cr
            # Depth of the deepest *detectable band*, not the deepest excursion.
            # A mineral whose real absorption sits just outside the range leaves
            # only a monotonic bow inside it, and max(1 - CR) reads that bow as a
            # deep feature. Judging degeneracy on it would wrongly mark such a
            # mineral as uniquely identifiable here when in truth this range sees
            # nothing but a sloped continuum.
            found = featmod.find_bands(wl, cr, min_depth=0.008)
            d = max((b.depth for b in found), default=0.0)
            if d > best_depth:
                best_profile, best_depth, best_spec = absorption, d, spec
        profiles.append(best_profile)
        depths[name] = best_depth
        albedos[name] = float(np.mean(best_spec))

    P = np.vstack(profiles)
    n = len(classes)

    # Pairwise distance.
    norms = np.linalg.norm(P, axis=1)
    D = np.zeros((n, n), dtype=np.float64)
    log_alb = np.log10(np.clip([albedos[c] for c in classes], 1e-4, None))

    for i in range(n):
        for j in range(i + 1, n):
            flat_i = depths[classes[i]] < FEATURELESS_DEPTH
            flat_j = depths[classes[j]] < FEATURELESS_DEPTH
            if flat_i and flat_j:
                ang = 0.0                      # both featureless: shape says nothing
            elif flat_i != flat_j:
                ang = 90.0                     # one has bands, the other does not
            else:
                den = norms[i] * norms[j]
                cos = (P[i] @ P[j]) / den if den > 1e-12 else 0.0
                ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            d = ang + ALBEDO_WEIGHT * abs(log_alb[i] - log_alb[j])
            D[i, j] = D[j, i] = d

    if n < 2:
        labels = np.array([0])
    else:
        Z = linkage(squareform(D, checks=False), method="average")
        labels = fcluster(Z, t=threshold, criterion="distance")

    label_map = {c: int(l) for c, l in zip(classes, labels)}
    members: dict = {}
    for c, l in label_map.items():
        members.setdefault(int(l), []).append(c)
    for l in members:
        members[l].sort(key=lambda m: -depths.get(m, 0.0))

    lookup = (library.usgs_group if hasattr(library, "usgs_group")
              else library.mineral)
    names = {l: _group_name(ms, depths, albedos, lookup)
             for l, ms in members.items()}

    return DegeneracyMap(range_nm=(float(wl.min()), float(wl.max())),
                         labels=label_map, members=members, names=names,
                         depths=depths, albedos=albedos)


_cache: dict = {}


def get_map(library, lo_nm: float, hi_nm: float) -> DegeneracyMap:
    key = (round(lo_nm, 1), round(hi_nm, 1), id(library))
    if key not in _cache:
        _cache[key] = build(library, lo_nm, hi_nm)
    return _cache[key]
