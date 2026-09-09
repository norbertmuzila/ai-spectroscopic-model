"""
The analysis engine.

One call in - a raw spectrum - and one structured result out. The stages:

    1.  Preprocess          counts -> reflectance -> continuum-removed
    2.  Quality control     gate the analysis on SNR, saturation, references
    3.  Physics matching    SAM / SCM / SID / band matching against the library
    4.  Learned inference   ensemble classifier over augmented training data
    5.  Fusion              combine the two in log space
    6.  Calibration         temperature scaling, so confidence means something
    7.  Conformal set       a prediction set with a coverage guarantee
    8.  Unmixing            abundances for the phases actually present
    9.  Range audit         which conclusions the instrument could physically support
    10. Interpretation      the evidence, in words, band by band

Stage 9 is the one that makes this trustworthy rather than merely confident. It
compares each candidate's diagnostic band positions against the measured range,
and if the evidence that would distinguish that mineral lies outside the
instrument's reach, the result says so explicitly instead of quietly presenting
a similarity score as an identification.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend import config as cfg_mod
from backend.library import store as libstore
from backend.models import calibration as calmod
from backend.models import classifier as clfmod
from backend.models import degeneracy as degmod
from backend.models import matcher as matchmod
from backend.models import unmixing as unmixmod
from backend.processing import features as featmod
from backend.processing import preprocess as prep
from backend.processing import quality as qcmod


@dataclass
class AnalysisResult:
    analysis_id: str
    timestamp: float
    sample_label: str
    spectrum: dict
    quality: dict
    identification: dict
    candidates: list
    prediction_set: dict
    composition: dict
    bands: list
    range_audit: dict
    interpretation: list
    engine: dict
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "sample_label": self.sample_label,
            "spectrum": self.spectrum,
            "quality": self.quality,
            "identification": self.identification,
            "candidates": self.candidates,
            "prediction_set": self.prediction_set,
            "composition": self.composition,
            "bands": self.bands,
            "range_audit": self.range_audit,
            "interpretation": self.interpretation,
            "engine": self.engine,
            "metadata": self.metadata,
        }


class AnalysisEngine:
    def __init__(self, load_model: bool = True):
        self.cfg = cfg_mod.load_config()
        self.grid = cfg_mod.analysis_grid()
        self.library = libstore.get_library()
        self.matcher = matchmod.SpectralMatcher(self.library)

        self.classifier: clfmod.EnsembleClassifier | None = None
        self.calibration = calmod.CalibrationBundle()
        self.model_metrics: dict = {}
        self.model_loaded = False
        if load_model:
            self.load_model()

        # Per-mineral reliability measured by leave-one-out against USGS ground
        # truth. Attached to every identification so a result carries its own
        # track record: "Hematite" from a mineral validated at 85% is a very
        # different claim from one the validation never once got right, and the
        # operator should not have to go looking for that distinction.
        self.validation: dict = {}
        try:
            vpath = self.cfg.resolve("model.dir") / "usgs_validation.json"
            if vpath.exists():
                import json
                self.validation = json.loads(vpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[engine] could not load USGS validation: {exc}")

        self.instrument_range = (
            float(self.cfg.get_path("instrument.wavelength_min_nm", 350.0)),
            float(self.cfg.get_path("instrument.wavelength_max_nm", 1000.0)),
        )
        self._counter = 0

    # ------------------------------------------------------------------
    def model_dir(self) -> Path:
        return self.cfg.resolve("model.dir")

    def load_model(self) -> bool:
        mdir = self.model_dir()
        clf_path = mdir / "ensemble.joblib"
        cal_path = mdir / "calibration.npz"
        met_path = mdir / "metrics.json"
        if not clf_path.exists():
            self.model_loaded = False
            return False
        try:
            self.classifier = clfmod.EnsembleClassifier.load(clf_path)
            if cal_path.exists():
                self.calibration = calmod.CalibrationBundle.load(cal_path)
            if met_path.exists():
                import json
                self.model_metrics = json.loads(met_path.read_text(encoding="utf-8"))
            self.model_loaded = True
        except Exception as exc:
            print(f"[engine] could not load model: {exc}")
            self.model_loaded = False
        return self.model_loaded

    # ------------------------------------------------------------------
    def _fuse(self, match_probs: dict, clf_probs: dict) -> dict:
        """
        Combine physics and learned evidence in log space.

        A geometric (log-linear) pool rather than an arithmetic average: it
        requires both sources to support a hypothesis, so a mineral the matcher
        rules out cannot be resurrected by the classifier alone. That is the
        conservative direction, and the right one when a wrong mineral ID is
        more costly than an uncertain one.
        """
        w_m = float(self.cfg.get_path("model.weight_matcher", 0.45))
        w_c = float(self.cfg.get_path("model.weight_classifier", 0.55))
        if not clf_probs:
            return dict(match_probs)
        if not match_probs:
            return dict(clf_probs)

        names = set(match_probs) | set(clf_probs)
        fused = {}
        for n in names:
            pm = max(match_probs.get(n, 1e-6), 1e-6)
            pc = max(clf_probs.get(n, 1e-6), 1e-6)
            fused[n] = w_m * np.log(pm) + w_c * np.log(pc)
        mx = max(fused.values())
        exp = {n: float(np.exp(v - mx)) for n, v in fused.items()}
        total = sum(exp.values())
        return {n: v / total for n, v in sorted(exp.items(), key=lambda kv: -kv[1])}

    # ------------------------------------------------------------------
    @staticmethod
    def _signal_range(wl_src, white, dark, lo, hi, threshold=0.06):
        """
        Narrow [lo, hi] to the span where the white reference carries real
        signal - at least ``threshold`` of its own peak.

        Without this the 350-380 nm and 990-1000 nm channels dominate the band
        detector with noise spikes, because reflectance there is a ratio of two
        near-zero numbers.
        """
        if white is None:
            return lo, hi, False
        w = np.asarray(white, dtype=np.float64)
        if dark is not None:
            w = w - np.asarray(dark, dtype=np.float64)
        if w.size != wl_src.size or not np.isfinite(w).any():
            return lo, hi, False
        peak = float(np.percentile(w, 99.5))
        if peak <= 0:
            return lo, hi, False
        good = w >= threshold * peak
        if good.sum() < 20:
            return lo, hi, False
        wl_good = wl_src[good]
        new_lo = max(lo, float(wl_good.min()))
        new_hi = min(hi, float(wl_good.max()))
        if new_hi - new_lo < 100.0:
            return lo, hi, False
        return new_lo, new_hi, (new_lo > lo + 0.5 or new_hi < hi - 0.5)

    # ------------------------------------------------------------------
    def _stray_light_floor(self, values, dark, white, already_reflectance) -> float:
        """
        Smallest band depth that can be believed, given how much signal the
        sample returned relative to the white reference.
        """
        self._last_signal_fraction = 1.0
        if already_reflectance or white is None:
            return 0.0
        s = np.asarray(values, dtype=np.float64)
        w = np.asarray(white, dtype=np.float64)
        d = np.asarray(dark, dtype=np.float64) if dark is not None else np.zeros_like(s)
        if s.shape != w.shape:
            return 0.0
        sig = float(np.median(s - d))
        ref = float(np.median(w - d))
        if ref <= 0.0:
            return 0.0
        frac = max(sig / ref, 1e-4)
        self._last_signal_fraction = frac
        stray = float(self.cfg.get_path("instrument.stray_light_fraction", 0.002))
        return float(np.clip(stray / frac, 0.0, 0.5))

    # ------------------------------------------------------------------
    def _reliability(self, name: str) -> dict | None:
        """
        How often leave-one-out validation against USGS got this mineral right.

        None means the mineral was never tested - which happens when the USGS
        library holds only one sample of it, so hiding that sample leaves nothing
        to find. That is a real gap in the evidence and is reported as such
        rather than as a passing grade.
        """
        per = (self.validation or {}).get("per_mineral") or {}
        rec = per.get(name)
        if not rec or not rec.get("trials"):
            return None
        hits, trials = rec["hits"], rec["trials"]
        return {
            "accuracy": round(hits / trials, 3),
            "trials": trials,
            "source": "leave-one-out vs USGS splib05a",
        }

    # ------------------------------------------------------------------
    def _range_audit(self, ranked: list, lo: float, hi: float) -> dict:
        """Report what the measured range can and cannot decide."""
        blind = []
        for name, _ in ranked[:10]:
            m = self.library.mineral(name)
            if m is None:
                continue
            cov = self.matcher.diagnostic_coverage(name, lo, hi)
            if cov < 1.0:
                outside = [round(c * 1000.0, 0) for c in m.diagnostic_centres
                           if not (lo <= c * 1000.0 <= hi)]
                blind.append({
                    "mineral": name,
                    "diagnostic_coverage": round(cov, 2),
                    "bands_outside_range_nm": outside,
                })

        total_known = len(self.library.classes)
        fully = sum(1 for c in self.library.classes
                    if self.matcher.diagnostic_coverage(c, lo, hi) >= 0.999)
        partly = sum(1 for c in self.library.classes
                     if 0.0 < self.matcher.diagnostic_coverage(c, lo, hi) < 0.999)

        return {
            "measured_range_nm": [round(lo, 1), round(hi, 1)],
            "library_minerals": total_known,
            "fully_diagnosable_here": fully,
            "partially_diagnosable_here": partly,
            "not_diagnosable_here": total_known - fully - partly,
            "candidates_with_evidence_outside_range": blind,
        }

    # ------------------------------------------------------------------
    def _interpret(self, top_name: str, confidence: float, bands: list,
                   pset: dict, audit: dict, qc: qcmod.QualityReport,
                   comp: unmixmod.UnmixResult, degmap=None) -> list:
        lines = []
        m = self.library.mineral(top_name)

        # Degeneracy first: if this instrument cannot separate the candidate
        # from its spectral twins, that governs how everything below reads.
        siblings = degmap.siblings(top_name) if degmap else []
        if siblings:
            depth = degmap.depths.get(top_name, 0.0)
            shown = ", ".join(siblings[:9])
            more = f", and {len(siblings) - 9} others" if len(siblings) > 9 else ""
            if depth < degmod.FEATURELESS_DEPTH:
                lines.append(
                    f"This sample has no usable absorption band in the measured range "
                    f"(deepest feature {depth * 100:.1f}%). Within {audit['measured_range_nm'][0]:.0f}-"
                    f"{audit['measured_range_nm'][1]:.0f} nm it is spectrally identical to "
                    f"{len(siblings) + 1} phases: {top_name}, {shown}{more}. The albedo and "
                    f"continuum slope narrow it to this group, but nothing in this "
                    f"wavelength range can subdivide it - SWIR coverage (1400-2500 nm) is "
                    f"required, where these minerals differ sharply.")
            else:
                lines.append(
                    f"{top_name} cannot be separated here from {shown}{more}: their "
                    f"absorption profiles differ by less than this measurement can "
                    f"resolve within {audit['measured_range_nm'][0]:.0f}-"
                    f"{audit['measured_range_nm'][1]:.0f} nm.")

        if bands:
            found = ", ".join(f"{b.centre_nm:.0f} nm (depth {b.depth * 100:.1f}%)"
                              for b in sorted(bands, key=lambda b: -b.depth)[:4])
            lines.append(f"Absorption features detected at {found}.")
        else:
            lines.append("No significant absorption bands were detected; the "
                         "identification rests on continuum shape and albedo alone, "
                         "which is weaker evidence.")

        if m is not None:
            lo, hi = audit["measured_range_nm"]
            inside = [c * 1000.0 for c in m.diagnostic_centres if lo <= c * 1000.0 <= hi]
            outside = [c * 1000.0 for c in m.diagnostic_centres
                       if not (lo <= c * 1000.0 <= hi)]
            if inside:
                matched = []
                for c in inside:
                    near = min((abs(b.centre_nm - c) for b in bands), default=1e9)
                    if near < 30.0:
                        matched.append(f"{c:.0f} nm")
                if matched:
                    lines.append(
                        f"{top_name} is expected to absorb at {', '.join(f'{c:.0f} nm' for c in inside)}; "
                        f"features were observed at {', '.join(matched)}, which supports the match.")
                else:
                    lines.append(
                        f"{top_name} should absorb at {', '.join(f'{c:.0f} nm' for c in inside)}, "
                        f"but no matching feature was measured there. Treat this identification "
                        f"with caution.")
            if outside:
                lines.append(
                    f"The strongest diagnostic features of {top_name} lie at "
                    f"{', '.join(f'{c:.0f} nm' for c in outside)}, outside the "
                    f"{lo:.0f}-{hi:.0f} nm range this instrument measures. That evidence "
                    f"could not be tested here.")
            if m.notes:
                lines.append(m.notes)

        pset_names = pset.get("minerals", [])
        groups = pset.get("groups", [])
        if len(pset_names) > 8 and groups:
            # Reported by spectral group rather than as a flat list: 30 mineral
            # names is noise, 3 groups is a finding.
            desc = "; ".join(
                f"{g['label']}" + (f" ({g['n_in_group']} phases)" if g["n_in_group"] > 1 else "")
                for g in groups[:5])
            lines.append(
                f"At the {pset.get('coverage', 0.95) * 100:.0f}% confidence level the data is "
                f"consistent with {len(pset_names)} minerals, falling into "
                f"{len(groups)} spectrally distinguishable group"
                f"{'s' if len(groups) != 1 else ''}: {desc}. A set this broad means the "
                f"measurement is under-determined - normally the diagnostic features lie "
                f"outside this instrument's range, or SNR is too low. That is the correct "
                f"answer to this measurement, not a model failure.")
        elif len(pset_names) > 1:
            lines.append(
                f"At the {pset.get('coverage', 0.95) * 100:.0f}% confidence level the data "
                f"cannot separate: {', '.join(pset_names)}. Distinguishing them would need "
                f"either a higher-SNR measurement or SWIR coverage.")
        elif len(pset_names) == 1:
            lines.append(
                f"The {pset.get('coverage', 0.95) * 100:.0f}% prediction set contains only "
                f"{pset_names[0]}, so the identification is unambiguous at that level.")

        if comp.n_endmembers > 1:
            parts = ", ".join(f"{k} {v * 100:.0f}%" for k, v in comp.abundances.items())
            lines.append(
                f"Unmixing ({comp.mode} model, RMSE {comp.rmse:.4f}) resolves the sample "
                f"into {parts}.")

        for w in qc.warnings:
            lines.append(f"Measurement note: {w}")
        for e in qc.errors:
            lines.append(f"Quality failure: {e}")

        return lines

    # ==================================================================
    def analyze(self, *,
                wavelength_nm: np.ndarray,
                values: np.ndarray,
                dark: np.ndarray | None = None,
                white: np.ndarray | None = None,
                already_reflectance: bool = False,
                nonlinearity_coeffs=None,
                sample_label: str = "Unlabelled sample",
                reference_age_minutes: float | None = None,
                metadata: dict | None = None) -> AnalysisResult:
        t_start = time.time()
        self._counter += 1
        analysis_id = f"A{int(t_start)}-{self._counter:04d}"

        wl_src = np.asarray(wavelength_nm, dtype=np.float64)
        lo = max(float(wl_src.min()), self.instrument_range[0], float(self.grid.min()))
        hi = min(float(wl_src.max()), self.instrument_range[1], float(self.grid.max()))

        # Trim to the range where the instrument actually has signal. At the
        # blue and red extremes the lamp output, grating efficiency and CCD
        # quantum efficiency all collapse together, so the white reference goes
        # to nearly zero and the reflectance ratio divides noise by noise. Those
        # channels produce convincing-looking absorption bands that are pure
        # amplified noise, so they are excluded rather than smoothed over.
        lo, hi, trimmed = self._signal_range(wl_src, white, dark, lo, hi)
        gmask = (self.grid >= lo) & (self.grid <= hi)
        if gmask.sum() < 20:
            gmask = np.ones_like(self.grid, dtype=bool)
            lo, hi = float(self.grid.min()), float(self.grid.max())
        wl = self.grid[gmask]

        # 1. Preprocess ------------------------------------------------
        ps = prep.process(wl_src, values, wl,
                          dark=dark, white=white,
                          already_reflectance=already_reflectance,
                          nonlinearity_coeffs=nonlinearity_coeffs)

        # 2. Quality ---------------------------------------------------
        qc = qcmod.assess(
            ps.raw_counts if not already_reflectance else None,
            wl, ps.reflectance,
            saturation_counts=float(self.cfg.get_path("instrument.saturation_counts", 65535)),
            saturation_warn=float(self.cfg.get_path("instrument.saturation_fraction_warn", 0.9)),
            min_snr=float(self.cfg.get_path("acquisition.min_snr_for_identification", 25.0)),
            instrument_range_nm=self.instrument_range,
            reference_age_minutes=reference_age_minutes,
            reference_ttl_minutes=float(self.cfg.get_path("acquisition.reference_ttl_minutes", 30)),
            had_white_reference=(white is not None) or already_reflectance,
        )

        if trimmed:
            qc.warnings.append(
                f"Analysis range narrowed to {lo:.0f}-{hi:.0f} nm: outside this span the "
                f"white reference falls below 6% of its peak, so reflectance there is "
                f"noise divided by noise and would generate false absorption bands.")

        # Band detection is gated on this measurement's own noise floor, so a
        # noisy dark sample cannot manufacture absorption features out of nothing.
        # Two independent noise estimates, and the larger wins: the propagated
        # photon-noise figure from preprocessing, and a direct measurement of the
        # residual wiggle left in the continuum-removed spectrum. They catch
        # different things, and under-estimating here is what lets a featureless
        # dark sample produce a confident identification.
        noise_sigma = max(ps.noise_sigma,
                          featmod.estimate_noise(ps.continuum_removed))

        # Stray-light floor. A spectrometer scatters a small fraction of total
        # flux into the wrong channels, and that contribution does NOT cancel in
        # R = (S-D)/(W-D): it is roughly constant while the sample signal is not.
        # On a dark sample returning only a few percent of the white reference,
        # the residual imprints a reproducible pseudo-absorption pattern several
        # percent deep - not noise, so no amount of averaging removes it, and
        # repeated measurements agree with each other while all being wrong.
        # Its amplitude is predictable: stray fraction divided by the fraction of
        # white-reference signal the sample returns. Bands shallower than that
        # are instrument artefact and must not drive an identification.
        stray_floor = self._stray_light_floor(values, dark, white, already_reflectance)
        bands = featmod.find_bands(wl, ps.continuum_removed,
                                   noise_sigma=noise_sigma,
                                   min_depth=max(0.012, stray_floor))
        if stray_floor > 0.02:
            qc.warnings.append(
                f"Sample returns only {100.0 * self._last_signal_fraction:.1f}% of the white "
                f"reference. At that level stray light distorts the spectral shape by about "
                f"{stray_floor * 100:.1f}%, so absorption bands shallower than that are "
                f"instrument artefact and have been excluded. Identification of very dark "
                f"phases rests on albedo and slope, not band position.")

        # 3. Physics matching -----------------------------------------
        has_white = (white is not None) or already_reflectance
        matches = self.matcher.match(wl, ps.reflectance, ps.continuum_removed,
                                     top_k=15, absolute_reflectance=has_white,
                                     noise_sigma=noise_sigma,
                                     min_band_depth=max(0.012, stray_floor))
        match_probs = self.matcher.scores_to_probabilities(matches)

        # 4. Learned inference ----------------------------------------
        clf_probs: dict = {}
        if self.classifier is not None:
            try:
                spec_vec, tab_vec = clfmod.encode(wl, ps.reflectance,
                                                  model_wl=self.classifier.wl)
                raw = self.classifier.predict_proba(spec_vec, tab_vec)
                clf_probs = {n: float(p) for n, p in zip(self.classifier.classes, raw)}
            except Exception as exc:
                print(f"[engine] classifier inference failed: {exc}")

        # 5-6. Fusion and calibration ---------------------------------
        fused = self._fuse(match_probs, clf_probs)
        names = list(fused.keys())
        probs = np.array([fused[n] for n in names], dtype=np.float64)
        if self.calibration.scaler.temperature != 1.0:
            probs = self.calibration.apply(probs[None, :])[0]
        order = np.argsort(-probs)
        ranked = [(names[i], float(probs[i])) for i in order]

        # 7. Conformal prediction set ---------------------------------
        conf_pred = self.calibration.conformal
        if conf_pred.qhat < 1.0 and conf_pred.n_calibration > 0:
            idxs, mass = conf_pred.predict_set(probs)
            pset_names = [names[i] for i in idxs]
            method_name = ("least-ambiguous set-valued classifier"
                           if conf_pred.method == "lac"
                           else "adaptive prediction sets")
            pset = {
                # The full set is what carries the coverage guarantee; truncating
                # it would void the guarantee, so the whole thing is reported and
                # the display layer decides how many to show.
                "minerals": pset_names,
                "size": len(pset_names),
                "coverage": round(1.0 - conf_pred.alpha, 3),
                "cumulative_probability": round(float(mass), 4),
                "method": f"conformal, {method_name}",
                "calibrated_on": conf_pred.n_calibration,
            }
        else:
            cum, pset_names = 0.0, []
            for n, p in ranked:
                pset_names.append(n)
                cum += p
                if cum >= 0.95 or len(pset_names) >= 5:
                    break
            pset = {
                "minerals": pset_names,
                "coverage": 0.95,
                "cumulative_probability": round(cum, 4),
                "method": "uncalibrated top-p (train the model for a conformal guarantee)",
                "calibrated_on": 0,
            }

        # 8. Unmixing --------------------------------------------------
        # Names and spectra are appended together: a candidate with no library
        # entry must not shift the indices, or the seed and the reported
        # abundances would silently attach to the wrong mineral.
        cand_names, cand_rows = [], []
        cols = np.flatnonzero(gmask)
        for n, _ in ranked[:8]:
            idx = self.library.entries_for(n)
            if idx.size == 0:
                continue
            cand_names.append(n)
            cand_rows.append(np.median(self.library.spectra[np.ix_(idx, cols)], axis=0))
        if cand_rows:
            # Seed with the identified mineral so the composition table and the
            # headline identification cannot disagree.
            comp = unmixmod.unmix(
                ps.reflectance, np.vstack(cand_rows), cand_names,
                max_endmembers=int(self.cfg.get_path("model.max_endmembers", 4)),
                seed_index=0)
        else:
            comp = unmixmod.UnmixResult({}, float("inf"), "areal", ps.reflectance.copy(), 0)

        # 9. Range audit + spectral degeneracy ------------------------
        audit = self._range_audit(ranked, lo, hi)
        degmap = degmod.get_map(self.library, lo, hi)
        pset["groups"] = degmap.collapse(pset["minerals"])
        pset["n_groups"] = len(pset["groups"])
        audit["degeneracy"] = degmap.summary()

        # ---- assemble identification --------------------------------
        top_name, top_p = ranked[0] if ranked else ("Unknown", 0.0)
        abstain_below = float(self.cfg.get_path("model.abstain_below", 0.35))
        coverage = self.matcher.diagnostic_coverage(top_name, lo, hi)
        top_match = next((m for m in matches if m.name == top_name), None)

        siblings = degmap.siblings(top_name)
        degenerate = len(siblings) > 0

        status = "identified"
        if not qc.passed:
            status = "rejected_quality"
        elif top_p < abstain_below and not degenerate:
            status = "inconclusive"
        elif degenerate:
            # The instrument cannot subdivide this mineral's spectral group, so
            # the group - not the individual mineral - is the real result.
            status = "degenerate"
        elif coverage < 0.5:
            status = "provisional_out_of_range"
        elif len(pset["minerals"]) > 1:
            status = "ambiguous"

        m_obj = self.library.mineral(top_name)
        # When the group cannot be subdivided, the group is the finding. Leading
        # with an arbitrary member of it - "Siderite, 22%" for what is actually a
        # bright featureless phase - reads as a specific claim the data does not
        # support, so the headline reports the group and the best-fitting member
        # is demoted to a secondary field.
        headline = degmap.label_for(top_name) if degenerate else top_name
        identification = {
            "status": status,
            "headline": headline if status != "rejected_quality" else None,
            "mineral": top_name if status != "rejected_quality" else None,
            "confidence": round(float(top_p), 4),
            "group": m_obj.group if m_obj else None,
            "formula": m_obj.formula if m_obj else None,
            "environments": list(m_obj.env) if m_obj else [],
            "diagnostic_coverage": round(float(coverage), 3),
            "reference_source": top_match.source if top_match else None,
            "reference_sample": top_match.sample_id if top_match else None,
            "sam_deg": round(top_match.sam_deg, 2) if top_match else None,
            "spectral_group": degmap.label_for(top_name),
            "indistinguishable_from": siblings,
            "band_depth_in_range": round(float(degmap.depths.get(top_name, 0.0)), 4),
            "usgs_validated": self._reliability(top_name),
        }

        interpretation = self._interpret(top_name, top_p, bands, pset, audit, qc,
                                         comp, degmap)

        step = float(np.median(np.diff(wl))) if wl.size > 1 else 1.0
        stride = max(1, int(round(1.0 / step)))
        spectrum = {
            "wavelength_nm": [round(float(v), 2) for v in wl[::stride]],
            "reflectance": [round(float(v), 5) for v in ps.reflectance[::stride]],
            "continuum": [round(float(v), 5) for v in ps.continuum[::stride]],
            "continuum_removed": [round(float(v), 5) for v in ps.continuum_removed[::stride]],
            "modelled": ([round(float(v), 5) for v in comp.modelled[::stride]]
                         if comp.n_endmembers else []),
            "range_nm": [round(lo, 1), round(hi, 1)],
        }

        return AnalysisResult(
            analysis_id=analysis_id,
            timestamp=t_start,
            sample_label=sample_label,
            spectrum=spectrum,
            quality=qc.as_dict(),
            identification=identification,
            candidates=[{"mineral": n,
                         "probability": round(p, 4),
                         "group": (self.library.mineral(n).group
                                   if self.library.mineral(n) else None),
                         "diagnostic_coverage": round(
                             self.matcher.diagnostic_coverage(n, lo, hi), 2)}
                        for n, p in ranked[:10]],
            prediction_set=pset,
            composition=comp.as_dict(),
            bands=[b.as_dict() for b in bands],
            range_audit=audit,
            interpretation=interpretation,
            engine={
                "model_loaded": self.model_loaded,
                "library": self.library.summary(),
                "fusion_weights": {
                    "matcher": float(self.cfg.get_path("model.weight_matcher", 0.45)),
                    "classifier": float(self.cfg.get_path("model.weight_classifier", 0.55)),
                },
                "temperature": round(self.calibration.scaler.temperature, 4),
                "validation": self.model_metrics.get("test", {}),
                "elapsed_ms": round((time.time() - t_start) * 1000.0, 1),
            },
            metadata=metadata or {},
        )


_engine: AnalysisEngine | None = None


def get_engine(reload: bool = False) -> AnalysisEngine:
    global _engine
    if _engine is None or reload:
        _engine = AnalysisEngine()
    return _engine
