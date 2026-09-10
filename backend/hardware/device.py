"""
Spectrometer device layer.

Talks to the Ocean Optics USB4000 through python-seabreeze. Two backends exist:
``cseabreeze`` (the vendor's C library) and ``pyseabreeze`` (pure Python over
libusb/PyUSB). Either works; pyseabreeze is easier to install on Windows because
it only needs a libusb driver bound to the device, which Zadig can do in a
minute.

Important operational detail: **the USB4000 can only be owned by one process at
a time.** If SpectraSuite or OceanView is running and connected, this layer
cannot open the device, and vice versa. That is a USB driver-binding constraint,
not something software can work around. Set
``spectrasuite.exclusive_device_mode: true`` in the config to run in
SpectraSuite-driven mode, where spectra arrive as exported files instead.

Reference handling is enforced here rather than left to the caller: a dark or
white reference taken at a different integration time is silently invalid, and
that is one of the most common sources of wrong reflectance values, so
``take_reference`` records the integration time and ``reference_status``
reports any mismatch.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Reference:
    counts: np.ndarray
    integration_time_ms: float
    scans_averaged: int
    timestamp: float = field(default_factory=time.time)

    @property
    def age_minutes(self) -> float:
        return (time.time() - self.timestamp) / 60.0


class DeviceError(RuntimeError):
    pass


class BaseSpectrometer:
    """Common interface shared by the real device and the simulator."""

    model = "generic"
    serial = "unknown"
    is_simulated = False

    def __init__(self):
        self._lock = threading.RLock()
        self.integration_time_ms = 100.0
        self.scans_to_average = 10
        self.boxcar_width = 2
        self.dark: Reference | None = None
        self.white: Reference | None = None
        self.nonlinearity_coeffs = None

    # ---- to implement ----------------------------------------------------
    def wavelengths(self) -> np.ndarray:
        raise NotImplementedError

    def _read_raw(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ---- settings --------------------------------------------------------
    def set_integration_time(self, ms: float) -> None:
        with self._lock:
            self.integration_time_ms = float(np.clip(ms, 0.01, 10000.0))
            self._apply_integration_time()

    def _apply_integration_time(self) -> None:
        pass

    def set_averaging(self, scans: int) -> None:
        self.scans_to_average = int(np.clip(scans, 1, 200))

    def set_boxcar(self, width: int) -> None:
        self.boxcar_width = int(np.clip(width, 0, 20))

    # ---- acquisition -----------------------------------------------------
    def acquire(self, scans: int | None = None) -> np.ndarray:
        """
        Combine ``scans`` readings and apply the boxcar, returning counts.

        Frames are combined with a **sigma-clipped mean** rather than a plain
        average. A cosmic ray or a transient hot pixel dumps thousands of counts
        into a single channel of a single frame; a plain mean spreads 1/N of
        that into the result, and if it happens during a dark or white reference
        it then contaminates *every* subsequent measurement as a permanent fake
        absorption band. Clipping outliers across frames removes them entirely
        while keeping almost all of the sqrt(N) noise benefit that a median
        combine would throw away.
        """
        n = int(scans or self.scans_to_average)
        with self._lock:
            frames = [np.asarray(self._read_raw(), dtype=np.float64) for _ in range(n)]
        return self.combine(frames)

    def read_frame(self) -> np.ndarray:
        """One raw detector frame, uncombined and unsmoothed."""
        with self._lock:
            return np.asarray(self._read_raw(), dtype=np.float64)

    def combine(self, frames) -> np.ndarray:
        """
        The frame combine used by every measurement. The live view feeds its
        rolling window through this too, so it is the same arithmetic as an
        analysis rather than a look-alike.
        """
        n = len(frames)
        if n >= 3:
            stack = np.vstack(frames)
            med = np.median(stack, axis=0)
            mad = np.median(np.abs(stack - med), axis=0) * 1.4826
            # Floor the scale by photon noise so a genuinely quiet channel does
            # not get a near-zero tolerance and reject its own valid samples.
            scale = np.maximum(mad, np.sqrt(np.maximum(med, 1.0)))
            keep = np.abs(stack - med) <= 4.0 * scale
            counts = keep.sum(axis=0)
            avg = np.where(counts > 0,
                           np.where(keep, stack, 0.0).sum(axis=0) / np.maximum(counts, 1),
                           med)
        else:
            avg = np.mean(np.vstack(frames), axis=0)

        if self.boxcar_width > 0:
            k = 2 * self.boxcar_width + 1
            kernel = np.ones(k) / k
            avg = np.convolve(avg, kernel, mode="same")
        return avg

    # ---- references ------------------------------------------------------
    def take_reference(self, kind: str, scans: int | None = None) -> Reference:
        if kind not in ("dark", "white"):
            raise ValueError("kind must be 'dark' or 'white'")
        counts = self.acquire(scans)
        ref = Reference(counts=counts,
                        integration_time_ms=self.integration_time_ms,
                        scans_averaged=int(scans or self.scans_to_average))
        setattr(self, kind, ref)
        return ref

    def clear_reference(self, kind: str) -> None:
        setattr(self, kind, None)

    def reference_status(self) -> dict:
        out = {}
        for kind in ("dark", "white"):
            ref: Reference | None = getattr(self, kind)
            if ref is None:
                out[kind] = {"present": False}
                continue
            mismatch = abs(ref.integration_time_ms - self.integration_time_ms) > 1e-6
            out[kind] = {
                "present": True,
                "integration_time_ms": ref.integration_time_ms,
                "scans_averaged": ref.scans_averaged,
                "age_minutes": round(ref.age_minutes, 2),
                "integration_time_mismatch": mismatch,
            }
        return out

    def reference_age_minutes(self) -> float | None:
        ages = [r.age_minutes for r in (self.dark, self.white) if r is not None]
        return max(ages) if ages else None

    def info(self) -> dict:
        wl = self.wavelengths()
        return {
            "model": self.model,
            "serial": self.serial,
            "simulated": self.is_simulated,
            "pixels": int(wl.size),
            "wavelength_min_nm": round(float(wl.min()), 2),
            "wavelength_max_nm": round(float(wl.max()), 2),
            "integration_time_ms": self.integration_time_ms,
            "scans_to_average": self.scans_to_average,
            "boxcar_width": self.boxcar_width,
            "references": self.reference_status(),
        }


class SeabreezeSpectrometer(BaseSpectrometer):
    """Real Ocean Optics hardware via python-seabreeze."""

    def __init__(self, spec):
        super().__init__()
        self._spec = spec
        self.model = getattr(spec, "model", "USB4000")
        self.serial = getattr(spec, "serial_number", "unknown")
        self._wl = np.asarray(spec.wavelengths(), dtype=np.float64)
        try:
            limits = spec.integration_time_micros_limits
            self._int_limits_us = (float(limits[0]), float(limits[1]))
        except Exception:
            self._int_limits_us = (3800.0, 10_000_000.0)
        try:
            f = spec.f
            self.nonlinearity_coeffs = list(
                f.nonlinearity_coefficients.get_nonlinearity_coefficients())
        except Exception:
            self.nonlinearity_coeffs = None
        self._apply_integration_time()

    def wavelengths(self) -> np.ndarray:
        return self._wl

    def _apply_integration_time(self) -> None:
        us = self.integration_time_ms * 1000.0
        lo, hi = self._int_limits_us
        us = float(np.clip(us, lo, hi))
        self.integration_time_ms = us / 1000.0
        self._spec.integration_time_micros(us)

    def _read_raw(self) -> np.ndarray:
        # correct_dark_counts uses the USB4000's optically masked pixels to
        # subtract the electrical dark level on every frame. It is not a
        # substitute for a proper dark reference with the shutter closed, but
        # it does remove the drifting baseline between references.
        try:
            return self._spec.intensities(correct_dark_counts=True,
                                          correct_nonlinearity=False)
        except Exception:
            return self._spec.intensities()

    def close(self) -> None:
        try:
            self._spec.close()
        except Exception:
            pass


def open_hardware() -> BaseSpectrometer:
    """
    Open the first real spectrometer, or raise.

    Deliberately has no simulator fallback. Silently substituting the simulator
    when someone asked for hardware is the worst possible behaviour: the
    dashboard fills with plausible spectra and plausible mineral identifications
    that have nothing to do with the sample under the probe. Callers that want a
    fallback must ask for it explicitly.
    """
    try:
        from seabreeze.spectrometers import Spectrometer, list_devices
    except ImportError as exc:
        raise DeviceError(
            "the seabreeze library is not installed - run: "
            'python -m pip install "seabreeze[pyseabreeze]" libusb-package') from exc

    devices = list_devices()
    if not devices:
        raise DeviceError("no Ocean Optics spectrometer is connected")
    return SeabreezeSpectrometer(Spectrometer(devices[0]))


def open_device(prefer_simulator: bool = False, seed: int = 0,
                allow_fallback: bool = True) -> BaseSpectrometer:
    """
    Open a spectrometer.

    ``allow_fallback`` controls what happens when hardware is requested but
    unavailable: the default returns a simulator carrying ``open_error`` so
    scripts and self-tests keep working without the instrument, while the API
    passes ``allow_fallback=False`` so a failed connect surfaces to the operator
    as a failure rather than as fabricated data.
    """
    from backend.hardware.simulator import SimulatedUSB4000

    if prefer_simulator:
        return SimulatedUSB4000(seed=seed)

    try:
        return open_hardware()
    except Exception as exc:
        if not allow_fallback:
            raise
        sim = SimulatedUSB4000(seed=seed)
        sim.open_error = str(exc)
        return sim


def list_available() -> list:
    try:
        from seabreeze.spectrometers import list_devices
        return [{"model": d.model, "serial": d.serial_number} for d in list_devices()]
    except Exception:
        return []
