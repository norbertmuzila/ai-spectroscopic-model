"""
Physics-based spectral matching.

This is the half of the engine that does not learn anything. It compares the
measured spectrum directly against library endmembers using metrics from the
remote-sensing literature, each of which is insensitive to a different nuisance
variable:

    SAM  - Spectral Angle Mapper. The angle between the two spectra treated as
           vectors. Completely invariant to multiplicative scaling, so it does
           not care about illumination level, grain packing or how far the fibre
           optic sits from the sample. This is why it is the workhorse metric.
    SCM  - Spectral Correlation Mapper. SAM on mean-centred spectra, which also
           removes additive offset (stray light, imperfect dark subtraction).
    SID  - Spectral Information Divergence. Treats the spectra as probability
           distributions and takes the symmetric KL divergence. Sensitive to
           subtle shape differences that SAM's cosine can wash out.
    BAND - Explicit absorption-band matching: does the sample have a band where
           this mineral should have one, at the right position and depth? This
           is the metric a human spectroscopist actually uses, and it is the
           only one of the four that degrades gracefully when a mineral's
           diagnostic bands are outside the instrument range.

The four are combined into a single similarity, and crucially the combination
is *masked to the wavelength range the instrument actually measured*. A library
spectrum covering 0.35-2.5 um is truncated to the sample's range before
comparison, so gypsum is never matched on evidence the USB4000 cannot see.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.processing import features as featmod
from backend.processing import preprocess as prep

# Weight given to absolute albedo and continuum slope, versus band shape.
# CONTINUUM_W_FLOOR applies when the spectrum has strong, well-defined bands
# (shape is then far more informative than brightness); it rises toward
# CONTINUUM_W_FLOOR + CONTINUUM_W_SPAN as the spectrum becomes featureless and
# brightness becomes the only remaining evidence. CONTINUUM_DEPTH_SCALE sets how
# quickly that handover happens, in units of band depth.
#
# Values chosen by sweeping scripts/../sweep over a 34-case panel of single
# phases, spectral twins, minerals with no in-range evidence, and mixtures.
# 0.30 is a broad optimum: below it, featureless dark and featureless bright
# phases get confused with each other; above it, brightness starts overriding
# genuine band evidence and identification degrades.
CONTINUUM_W_FLOOR = 0.30
CONTINUUM_W_SPAN = 0.35
CONTINUUM_DEPTH_SCALE = 0.030


@dataclass
class MatchResult:
    name: str
    similarity: float          # 0-1, higher is better
    sam_deg: float
    correlation: float
    sid: float
    band_score: float
    best_entry: int
    source: str                # 'usgs' or 'synthetic'
    sample_id: str
    diagnostic_coverage: float  # fraction of diagnostic bands inside range

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "similarity": round(float(self.similarity), 4),
            "sam_deg": round(float(self.sam_deg), 3),
            "correlation": round(float(self.correlation), 4),
            "sid": round(float(self.sid), 5),
            "band_score": round(float(self.band_score), 4),
            "source": self.source,
            "reference_sample": self.sample_id,
            "diagnostic_coverage": round(float(self.diagnostic_coverage), 3),
        }


# ---------------------------------------------------------------------------
#  Metrics (vectorised over a library matrix)
# ---------------------------------------------------------------------------
def spectral_angle(sample: np.ndarray, library: np.ndarray) -> np.ndarray:
    """Angle in radians between the sample and every library row."""
    s = np.asarray(sample, dtype=np.float64)
    L = np.asarray(library, dtype=np.float64)
    num = L @ s
    den = np.linalg.norm(L, axis=1) * np.linalg.norm(s)
    den = np.where(den < 1e-12, 1e-12, den)
    return np.arccos(np.clip(num / den, -1.0, 1.0))


def spectral_correlation(sample: np.ndarray, library: np.ndarray) -> np.ndarray:
    s = np.asarray(sample, dtype=np.float64)
    L = np.asarray(library, dtype=np.float64)
    sc = s - s.mean()
    Lc = L - L.mean(axis=1, keepdims=True)
    num = Lc @ sc
    den = np.linalg.norm(Lc, axis=1) * np.linalg.norm(sc)
    den = np.where(den < 1e-12, 1e-12, den)
    return np.clip(num / den, -1.0, 1.0)


def spectral_information_divergence(sample: np.ndarray, library: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(sample, dtype=np.float64), 1e-9, None)
    L = np.clip(np.asarray(library, dtype=np.float64), 1e-9, None)
    p = s / s.sum()
    q = L / L.sum(axis=1, keepdims=True)
    kl_pq = np.sum(p * np.log(p / q), axis=1)
    kl_qp = np.sum(q * np.log(q / p), axis=1)
    return kl_pq + kl_qp


def band_match_score(sample_bands, ref_bands, tol_nm: float = 22.0) -> float:
    """
    Score how well two band lists agree on position and depth.

    Position agreement is weighted far more heavily than depth agreement,
    because band position is chemistry and band depth is mostly grain size.
    """
    if not sample_bands and not ref_bands:
        return 0.5          # both featureless - weak, non-committal agreement
    if not sample_bands or not ref_bands:
        return 0.0

    total_w = 0.0
    total_s = 0.0
    for rb in ref_bands:
        w = rb.depth
        total_w += w
        best = 0.0
        for sb in sample_bands:
            dpos = abs(sb.centre_nm - rb.centre_nm)
            if dpos > 4.0 * tol_nm:
                continue
            pos_score = float(np.exp(-0.5 * (dpos / tol_nm) ** 2))
            ddep = abs(sb.depth - rb.depth) / max(rb.depth, 1e-6)
            dep_score = float(np.exp(-0.5 * (ddep / 0.85) ** 2))
            best = max(best, 0.78 * pos_score + 0.22 * pos_score * dep_score)
        total_s += w * best

    recall = total_s / max(total_w, 1e-9)

    # Penalise strong sample bands the reference cannot explain: an unexplained
    # deep band is decisive evidence against a match.
    unexplained = 0.0
    sample_w = 0.0
    for sb in sample_bands:
        sample_w += sb.depth
        near = min((abs(sb.centre_nm - rb.centre_nm) for rb in ref_bands),
                   default=1e9)
        if near > 2.5 * tol_nm:
            unexplained += sb.depth
    precision = 1.0 - unexplained / max(sample_w, 1e-9)

    return float(np.clip(0.62 * recall + 0.38 * precision, 0.0, 1.0))


# ---------------------------------------------------------------------------
#  Matcher
# ---------------------------------------------------------------------------
class SpectralMatcher:
    def __init__(self, library, weights: dict | None = None):
        self.library = library
        self.weights = weights or {"sam": 0.34, "scm": 0.24, "sid": 0.12, "band": 0.30}
        self._cr_cache: np.ndarray | None = None
        self._bands_cache: list | None = None
        self._cache_key = None
        self._usgs_bands: dict | None = None

    # -- library-side precomputation --------------------------------------
    def _prepare(self, mask: np.ndarray):
        """Continuum-remove every library entry over the instrument's range."""
        key = (int(mask.sum()), int(np.flatnonzero(mask)[0]) if mask.any() else -1)
        if self._cache_key == key and self._cr_cache is not None:
            return self._cr_cache, self._bands_cache

        wl = self.library.grid_nm[mask]
        lib = self.library.spectra[:, mask]
        cr = np.empty_like(lib)
        bands = []
        for i in range(lib.shape[0]):
            _, cr_i = prep.continuum_removal(wl, lib[i])
            cr[i] = cr_i
            bands.append(featmod.find_bands(wl, cr_i))

        self._cr_cache, self._bands_cache, self._cache_key = cr, bands, key
        return cr, bands

    def diagnostic_coverage(self, name: str, lo_nm: float, hi_nm: float) -> float:
        """
        Fraction of a mineral's *diagnostic* absorption bands that fall inside
        the measured wavelength range.

        This is the honesty valve of the whole system. Gypsum's diagnostic bands
        are at 1.45, 1.49 and 1.94 um; against a 350-1000 nm measurement its
        coverage is 0.0, and no amount of curve similarity is allowed to turn
        that into a confident identification.
        """
        centres = self._diagnostic_centres_nm(name)
        if centres is None:
            return 1.0          # nothing known about this mineral's bands
        if not centres:
            return 0.0          # measured, and it has no bands at all
        inside = sum(1 for c in centres if lo_nm <= c <= hi_nm)
        return inside / len(centres)

    def _diagnostic_centres_nm(self, name: str):
        """
        Band centres in nanometres, measured from the USGS library where it has
        this mineral, otherwise from the hand-written knowledge base.

        Returning None means "unknown" and returning an empty list means
        "measured, and there are none" - two very different statements that a
        single empty list would conflate, with the effect that every mineral
        would pass the coverage check.
        """
        if self._usgs_bands is None:
            try:
                from backend.library import usgs_splib
                self._usgs_bands = usgs_splib.diagnostic_bands()
            except Exception:
                self._usgs_bands = {}
        if name in self._usgs_bands:
            return [b["centre_nm"] for b in self._usgs_bands[name]]
        m = self.library.mineral(name)
        if m is None:
            return None
        return [c * 1000.0 for c in m.diagnostic_centres] or None

    # -- main entry point --------------------------------------------------
    @staticmethod
    def _albedo_slope(wl: np.ndarray, spectra: np.ndarray):
        """Mean reflectance and ln-reflectance slope per micrometre."""
        r = np.clip(np.atleast_2d(spectra), 1e-5, None)
        x = wl / 1000.0
        xc = x - x.mean()
        ln_r = np.log(r)
        slope = (ln_r - ln_r.mean(axis=1, keepdims=True)) @ xc / max(float(xc @ xc), 1e-12)
        return r.mean(axis=1), slope

    def match(self, wl_nm: np.ndarray, reflectance: np.ndarray,
              continuum_removed: np.ndarray, top_k: int = 12,
              absolute_reflectance: bool = True,
              noise_sigma: float | None = None,
              min_band_depth: float = 0.012,
              exclude_entries: set | None = None) -> list:
        grid = self.library.grid_nm
        lo, hi = float(wl_nm.min()), float(wl_nm.max())
        mask = (grid >= lo - 1e-6) & (grid <= hi + 1e-6)
        if mask.sum() < 10:
            return []

        wl_m = grid[mask]
        sample_cr = np.interp(wl_m, wl_nm, continuum_removed)
        sample_r = np.interp(wl_m, wl_nm, reflectance)
        # Library entries are noise-free, so they keep the fixed floor; the
        # measured sample is held to its own noise level.
        sample_bands = featmod.find_bands(wl_m, sample_cr, noise_sigma=noise_sigma,
                                          min_depth=min_band_depth)

        lib_cr, lib_bands = self._prepare(mask)
        lib_r = self.library.spectra[:, mask]

        # Shape metrics run on the continuum-removed spectra: this is where the
        # mineralogy lives, with albedo and slope divided out.
        sam = spectral_angle(1.0 - sample_cr, 1.0 - lib_cr)
        scm = spectral_correlation(sample_cr, lib_cr)
        sid = spectral_information_divergence(sample_cr, lib_cr)

        sam_s = np.exp(-sam / 0.22)
        scm_s = np.clip((scm + 1.0) / 2.0, 0.0, 1.0) ** 2
        sid_s = np.exp(-sid / 0.06)

        w = self.weights
        n = lib_cr.shape[0]
        band_s = np.zeros(n)
        # Band matching is the expensive metric, so only run it on entries the
        # cheap metrics have not already ruled out.
        prelim = w["sam"] * sam_s + w["scm"] * scm_s + w["sid"] * sid_s
        shortlist = np.argsort(-prelim)[:max(top_k * 8, 120)]
        for i in shortlist:
            band_s[i] = band_match_score(sample_bands, lib_bands[i])

        shape = (w["sam"] * sam_s + w["scm"] * scm_s
                 + w["sid"] * sid_s + w["band"] * band_s) / sum(w.values())

        # Albedo and continuum slope. Continuum removal deliberately discards
        # both, which is right when the mineral has absorption bands - but when
        # it has none, they are the *only* information left. A featureless dark
        # opaque (magnetite, ilmenite) and a featureless bright evaporite
        # (gypsum, halite) have identical continuum-removed spectra and differ
        # by a factor of twenty in reflectance, so without this term the matcher
        # picks between them essentially at random.
        # Depth of the deepest *significant* band, not the deepest excursion -
        # otherwise noise alone makes a featureless spectrum look featured and
        # suppresses the albedo term exactly where it is needed most.
        sample_depth = max((b.depth for b in sample_bands), default=0.0)
        if absolute_reflectance:
            lib_alb, lib_slope = self._albedo_slope(wl_m, lib_r)
            s_alb, s_slope = self._albedo_slope(wl_m, sample_r[None, :])
            alb_s = np.exp(-np.abs(np.log10(np.maximum(lib_alb, 1e-5))
                                   - np.log10(max(float(s_alb[0]), 1e-5))) / 0.30)
            slope_s = np.exp(-np.abs(lib_slope - float(s_slope[0])) / 0.55)
            continuum_score = 0.65 * alb_s + 0.35 * slope_s
            w_cont = CONTINUUM_W_FLOOR + CONTINUUM_W_SPAN * float(
                np.exp(-(sample_depth / CONTINUUM_DEPTH_SCALE) ** 2))
        else:
            # Without a white reference the reflectance scale is arbitrary, so
            # absolute albedo means nothing and only the slope is usable.
            _, lib_slope = self._albedo_slope(wl_m, lib_r)
            _, s_slope = self._albedo_slope(wl_m, sample_r[None, :])
            continuum_score = np.exp(-np.abs(lib_slope - float(s_slope[0])) / 0.55)
            w_cont = 0.10

        total = (1.0 - w_cont) * shape + w_cont * continuum_score

        # Leave-one-out support: an excluded entry is removed from scoring
        # entirely, so a validation fold cannot match a spectrum against itself.
        # Without this every accuracy number would be a lookup, not a test.
        if exclude_entries:
            for e in exclude_entries:
                if 0 <= e < total.size:
                    total[e] = -np.inf

        # Collapse per-entry scores to per-mineral, keeping the best entry.
        names = np.asarray(self.library.names)
        results = []
        for name in self.library.classes:
            idx = np.flatnonzero(names == name)
            if idx.size == 0:
                continue
            if not np.isfinite(total[idx]).any():
                continue                 # every reference for this mineral was excluded
            best = idx[int(np.argmax(total[idx]))]
            results.append(MatchResult(
                name=name,
                similarity=float(total[best]),
                sam_deg=float(np.degrees(sam[best])),
                correlation=float(scm[best]),
                sid=float(sid[best]),
                band_score=float(band_s[best]),
                best_entry=int(best),
                source=self.library.sources[best],
                sample_id=self.library.sample_ids[best],
                diagnostic_coverage=self.diagnostic_coverage(name, lo, hi),
            ))

        results.sort(key=lambda r: -r.similarity)
        return results[:top_k]

    def scores_to_probabilities(self, results: list, temperature: float = 0.055,
                                coverage_floor: float = 0.10) -> dict:
        """
        Convert similarities to a probability distribution over minerals.

        Candidates whose diagnostic evidence lies outside the measured range are
        down-weighted before the softmax rather than after, so that the
        probability mass genuinely redistributes to hypotheses the instrument
        can actually test.
        """
        if not results:
            return {}
        names = [r.name for r in results]
        sims = np.array([r.similarity for r in results], dtype=np.float64)
        cov = np.array([r.diagnostic_coverage for r in results], dtype=np.float64)
        # Concave in coverage, so the penalty is a hammer at zero and barely a
        # nudge above a half. A mineral with none of its diagnostic bands in
        # range must be pushed hard down - there is no evidence for it at all.
        # But one with two of its four bands visible is genuinely supported, and
        # a linear penalty demoted such minerals below wrong answers that merely
        # happened to have full coverage, costing real accuracy for no gain in
        # honesty.
        penalty = coverage_floor + (1.0 - coverage_floor) * np.power(
            np.clip(cov, 0.0, 1.0), 0.4)
        logits = sims / max(temperature, 1e-6) + np.log(np.maximum(penalty, 1e-6))
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return dict(zip(names, (float(x) for x in p)))
