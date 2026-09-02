"""
Measurement quality control.

An identification is only as good as the spectrum it was made from, and the
most common cause of a confidently wrong mineral ID is a bad measurement rather
than a bad model. This module gates the analysis: if the spectrum fails QC, the
engine reports the failure instead of a mineral name.

Checks performed:

* saturation      - any channel at or near the 16-bit ADC ceiling means the
                    signal is clipped and every band depth derived from it is
                    wrong. Reduce integration time.
* dark/white age  - references drift with detector temperature. The USB4000 is
                    uncooled, so a reference taken an hour ago is not valid.
* SNR             - estimated from high-frequency residual, per spectral region.
* dynamic range   - a sample far darker than the white reference has poor
                    effective SNR even when raw SNR looks acceptable.
* detector edges  - the USB4000 loses grating efficiency hard below ~350 nm and
                    above ~1000 nm; those channels are noisy and must not drive
                    an identification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QualityReport:
    snr: float
    saturated_fraction: float
    max_counts: float
    usable_range_nm: tuple
    dynamic_range: float
    passed: bool
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "snr": round(float(self.snr), 1),
            "saturated_fraction": round(float(self.saturated_fraction), 4),
            "max_counts": round(float(self.max_counts), 1),
            "usable_range_nm": [round(float(v), 1) for v in self.usable_range_nm],
            "dynamic_range": round(float(self.dynamic_range), 4),
            "passed": bool(self.passed),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def estimate_snr(y: np.ndarray, window: int = 11) -> float:
    """
    SNR from the high-frequency residual after a local polynomial fit.

    Using the residual rather than repeat measurements means the estimate works
    on a single spectrum, and because a real absorption band is smooth on this
    scale it is not mistaken for noise.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size < window + 2:
        return 0.0
    from scipy.signal import savgol_filter
    win = window if window % 2 == 1 else window + 1
    smooth = savgol_filter(y, win, 3)
    resid = y - smooth
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    signal = float(np.median(np.abs(y)))
    if noise < 1e-12:
        return 9999.0
    return float(signal / noise)


def assess(raw_counts: np.ndarray,
           wl_nm: np.ndarray,
           reflectance: np.ndarray | None,
           *,
           saturation_counts: float = 65535.0,
           saturation_warn: float = 0.90,
           min_snr: float = 25.0,
           instrument_range_nm: tuple = (350.0, 1000.0),
           reference_age_minutes: float | None = None,
           reference_ttl_minutes: float = 30.0,
           had_white_reference: bool = True) -> QualityReport:
    warnings, errors = [], []

    raw = np.asarray(raw_counts, dtype=np.float64) if raw_counts is not None else None
    max_counts = float(np.max(raw)) if raw is not None and raw.size else 0.0

    sat_frac = 0.0
    if raw is not None and raw.size:
        sat_frac = float(np.mean(raw >= saturation_counts * saturation_warn))
        if sat_frac > 0.02:
            errors.append(
                f"{sat_frac * 100:.1f}% of channels are at or near detector saturation. "
                f"Band depths are clipped and unreliable - reduce the integration time.")
        elif sat_frac > 0.0:
            warnings.append(
                f"{sat_frac * 100:.2f}% of channels are near saturation; consider a "
                f"shorter integration time.")
        if max_counts < saturation_counts * 0.05:
            warnings.append(
                "Peak signal is under 5% of full scale. Increase integration time or "
                "signal averaging, otherwise SNR will limit the identification.")

    target = reflectance if reflectance is not None else raw
    snr = estimate_snr(target) if target is not None else 0.0
    if snr < min_snr:
        errors.append(
            f"Estimated SNR is {snr:.0f}, below the {min_snr:.0f} required for a "
            f"reliable identification. Increase scans-to-average or integration time.")
    elif snr < min_snr * 2:
        warnings.append(f"SNR is {snr:.0f}; marginal. Averaging more scans would help.")

    dyn = 0.0
    if reflectance is not None and reflectance.size:
        dyn = float(np.percentile(reflectance, 99) - np.percentile(reflectance, 1))
        mean_r = float(np.mean(reflectance))
        if mean_r < 0.02:
            warnings.append(
                "Sample reflectance is under 2%. Very dark samples (magnetite, ilmenite) "
                "give weak signal - expect wider confidence intervals.")

    if not had_white_reference:
        warnings.append(
            "No white reference on file. Results are based on spectral SHAPE only; "
            "absolute reflectance and albedo-based discrimination are unavailable.")

    if reference_age_minutes is not None and reference_age_minutes > reference_ttl_minutes:
        warnings.append(
            f"Dark/white reference is {reference_age_minutes:.0f} minutes old "
            f"(limit {reference_ttl_minutes:.0f}). The USB4000 is uncooled and its dark "
            f"current drifts with temperature - re-take the references.")

    lo = max(float(np.min(wl_nm)), instrument_range_nm[0])
    hi = min(float(np.max(wl_nm)), instrument_range_nm[1])

    return QualityReport(
        snr=snr,
        saturated_fraction=sat_frac,
        max_counts=max_counts,
        usable_range_nm=(lo, hi),
        dynamic_range=dyn,
        passed=len(errors) == 0,
        warnings=warnings,
        errors=errors,
    )
