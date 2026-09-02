"""
Physically-grounded USB4000 simulator.

This is not a stub that returns a sine wave. It models the actual signal chain
so that the preprocessing, QC, matching and calibration code is exercised
against realistic data:

    lamp spectral radiance (tungsten-halogen, Planck at ~2960 K)
      x  sample reflectance (from the mineral knowledge base)
      x  grating efficiency + detector quantum efficiency
      -> photoelectrons in each of 3648 pixels
      +  shot noise (Poisson), dark current (temperature dependent),
         read noise, fixed-pattern noise, stray light
      -> 16-bit ADC counts, saturating at 65535

The wavelength axis uses the USB4000's real third-order pixel-to-wavelength
polynomial form, so the resampling step is doing genuine work rather than a
no-op relabelling.

Set the loaded sample with ``load_sample('Hematite')`` or a mixture with
``load_mixture({'Olivine (Fo50)': 0.6, 'Augite (HCP)': 0.4})``.
"""
from __future__ import annotations

import time

import numpy as np

from backend.hardware.device import BaseSpectrometer
from backend.library import minerals as mindb
from backend.library import synth

PLANCK_H = 6.62607015e-34
LIGHT_C = 2.99792458e8
BOLTZ_K = 1.380649e-23


def _planck_radiance(wl_nm: np.ndarray, temperature_k: float) -> np.ndarray:
    wl_m = np.asarray(wl_nm, dtype=np.float64) * 1e-9
    a = 2.0 * PLANCK_H * LIGHT_C ** 2 / wl_m ** 5
    b = np.expm1(PLANCK_H * LIGHT_C / (wl_m * BOLTZ_K * temperature_k))
    return a / b


class SimulatedUSB4000(BaseSpectrometer):
    model = "USB4000 (simulated)"
    is_simulated = True

    def __init__(self, seed: int = 0, pixels: int = 3648,
                 wl_start: float = 344.0, wl_end: float = 1041.0):
        super().__init__()
        self.serial = f"SIM{seed:04d}"
        self.open_error: str | None = None
        self._rng = np.random.default_rng(seed)

        # Real USB4000 wavelength calibration is a cubic in pixel index; the
        # dispersion is noticeably non-uniform across the array.
        p = np.arange(pixels, dtype=np.float64)
        span = wl_end - wl_start
        self._wl = (wl_start
                    + span * (p / (pixels - 1))
                    - 0.045 * span * (p / (pixels - 1)) ** 2
                    + 0.030 * span * (p / (pixels - 1)) ** 3)

        # Instrument response: lamp x grating x CCD quantum efficiency. The
        # USB4000's response collapses below ~360 nm and above ~1000 nm, which
        # is exactly where noisy, untrustworthy channels come from.
        lamp = _planck_radiance(self._wl, 2960.0)
        lamp /= lamp.max()
        grating = np.exp(-0.5 * ((self._wl - 620.0) / 340.0) ** 2)
        qe = (1.0 / (1.0 + np.exp(-(self._wl - 368.0) / 13.0))) \
            * (1.0 / (1.0 + np.exp((self._wl - 985.0) / 21.0)))
        self._response = lamp * grating * qe
        self._response /= self._response.max()

        # Detector characteristics.
        self._fixed_pattern = 1.0 + self._rng.normal(0.0, 0.004, pixels)
        self._dark_offset = 1180.0 + self._rng.normal(0.0, 9.0, pixels)
        self._detector_temp_c = 28.0
        self._full_well_counts = 65535.0
        self._peak_counts_per_ms = 620.0

        self.nonlinearity_coeffs = [1.0, -2.1e-7, 5.4e-12, -3.9e-17,
                                    0.0, 0.0, 0.0, 0.0]

        self._sample_name = "Olivine (Fo50)"
        self._sample_reflectance = None
        self._mixture = {self._sample_name: 1.0}
        self._white_panel = 0.985      # Spectralon diffuse reflectance
        self._shutter_closed = False
        self.load_mixture(self._mixture)

    # ---- sample control --------------------------------------------------
    def load_sample(self, name: str, grain_size: float = 1.0,
                    weathering: float = 0.0) -> None:
        self.load_mixture({name: 1.0}, grain_size=grain_size, weathering=weathering)

    def load_mixture(self, fractions: dict, grain_size: float = 1.0,
                     weathering: float = 0.0, mode: str = "intimate") -> None:
        names, weights, specs = [], [], []
        for name, frac in fractions.items():
            m = mindb.BY_NAME.get(name)
            if m is None or frac <= 0:
                continue
            names.append(name)
            weights.append(float(frac))
            specs.append(synth.synthesize(m, self._wl, grain_size=grain_size,
                                          weathering=weathering, rng=self._rng))
        if not specs:
            raise ValueError("no valid minerals in mixture")
        arr = np.vstack(specs)
        w = np.asarray(weights)
        w = w / w.sum()
        self._sample_reflectance = synth.mix(arr, w, mode=mode)
        self._mixture = dict(zip(names, (float(x) for x in w)))
        self._sample_name = " + ".join(
            f"{n} {f * 100:.0f}%" for n, f in self._mixture.items())

    def set_shutter(self, closed: bool) -> None:
        """Closing the shutter is how a dark reference is taken."""
        self._shutter_closed = bool(closed)

    @property
    def loaded_sample(self) -> dict:
        return {"label": self._sample_name, "mixture": self._mixture}

    def available_samples(self) -> list:
        return [m.name for m in mindb.ALL_MINERALS]

    # ---- device interface ------------------------------------------------
    def wavelengths(self) -> np.ndarray:
        return self._wl

    def _read_raw(self) -> np.ndarray:
        t_int = self.integration_time_ms
        rng = self._rng

        # Dark current roughly doubles every 6-7 C on an uncooled CCD, and it
        # scales with integration time. This is what makes stale dark
        # references wrong, and the QC module exists to catch that.
        dark_rate = 0.9 * 2.0 ** ((self._detector_temp_c - 25.0) / 6.5)
        dark = self._dark_offset + dark_rate * t_int

        if self._shutter_closed:
            signal = np.zeros_like(self._wl)
        else:
            refl = self._sample_reflectance
            signal = self._response * refl * self._peak_counts_per_ms * t_int

        signal = signal * self._fixed_pattern

        # Stray light: a small fraction of total flux redistributed flat across
        # the array. It is why deep absorption bands never quite reach zero.
        stray = 0.0016 * float(np.mean(signal))
        signal = signal + stray

        total = signal + dark
        # Photon shot noise, then read noise.
        noisy = rng.poisson(np.clip(total, 0.0, 1e9)).astype(np.float64)
        noisy += rng.normal(0.0, 12.0, noisy.size)

        # Occasional cosmic-ray hit, so the despiker has something to do.
        if rng.random() < 0.03:
            hit = rng.integers(0, noisy.size)
            noisy[hit] += rng.uniform(2000.0, 20000.0)

        # Forward nonlinearity, then ADC saturation.
        frac = np.clip(noisy / self._full_well_counts, 0.0, 1.5)
        noisy = noisy * (1.0 - 0.035 * frac ** 2)

        time.sleep(min(t_int / 1000.0, 0.02))
        return np.clip(noisy, 0.0, self._full_well_counts)

    def take_reference(self, kind: str, scans: int | None = None):
        """
        The simulator models the physical act of referencing: dark closes the
        shutter, white swaps in a Spectralon panel.
        """
        if kind == "dark":
            self.set_shutter(True)
            try:
                return super().take_reference("dark", scans)
            finally:
                self.set_shutter(False)

        if kind == "white":
            saved = self._sample_reflectance
            self._sample_reflectance = np.full_like(self._wl, self._white_panel)
            try:
                return super().take_reference("white", scans)
            finally:
                self._sample_reflectance = saved

        return super().take_reference(kind, scans)

    def info(self) -> dict:
        base = super().info()
        base["loaded_sample"] = self.loaded_sample
        base["detector_temp_c"] = self._detector_temp_c
        if self.open_error:
            base["hardware_open_error"] = self.open_error
        return base
