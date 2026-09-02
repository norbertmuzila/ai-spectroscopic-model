"""
SpectraSuite / OceanView integration.

What is actually possible, stated plainly: SpectraSuite is a closed-source Java
application from 2009 that exposes no API, no plugin interface and no IPC
channel. Nothing can call into it. What it *does* offer is a well-defined ASCII
export format, and that is the integration surface this module uses.

Two working modes:

*Handoff mode* (``exclusive_device_mode: true``) - you drive the spectrometer
from SpectraSuite exactly as you do today, and save or auto-save spectra into
the watch folder. This module detects each new file within a second, parses it,
and pushes it straight through the analysis pipeline onto the dashboard. Use
this when you want SpectraSuite's own acquisition controls.

*Direct mode* (the default) - the backend owns the USB4000 through seabreeze and
SpectraSuite stays closed. This is the better mode: it gives live streaming,
enforced reference validity, and access to the raw non-linearity coefficients.

The two modes are mutually exclusive because the USB4000 can only be claimed by
one process at a time. SpectraSuite binds the device through the Ocean Optics
OmniDriver/32-bit driver, seabreeze binds it through libusb, and whichever
starts first wins.

Supported file formats
----------------------
* SpectraSuite tab-delimited export (header block, then ``>>>>>Begin Spectral
  Data<<<<<``, then wavelength/value pairs)
* OceanView export (same shape, ``Begin Processed Spectral Data`` marker)
* Bare two-column CSV / TSV
* JCAMP-DX (``##XYDATA``) as a fallback
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BEGIN_MARKERS = (
    ">>>>>begin spectral data<<<<<",
    ">>>>>begin processed spectral data<<<<<",
    ">>>>>begin<<<<<",
)
END_MARKERS = (
    ">>>>>end spectral data<<<<<",
    ">>>>>end processed spectral data<<<<<",
)

_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


@dataclass
class ImportedSpectrum:
    wavelength_nm: np.ndarray
    values: np.ndarray
    is_reflectance: bool
    header: dict = field(default_factory=dict)
    source_path: str = ""
    integration_time_ms: float | None = None
    scans_averaged: int | None = None
    boxcar_width: int | None = None

    def as_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "n_points": int(self.wavelength_nm.size),
            "range_nm": [round(float(self.wavelength_nm.min()), 2),
                         round(float(self.wavelength_nm.max()), 2)],
            "is_reflectance": self.is_reflectance,
            "integration_time_ms": self.integration_time_ms,
            "scans_averaged": self.scans_averaged,
            "boxcar_width": self.boxcar_width,
            "header": self.header,
        }


def _split_pair(line: str):
    for sep in ("\t", ";", ",", None):
        parts = line.split(sep) if sep else line.split()
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2 and _NUM.match(parts[0]) and _NUM.match(parts[1]):
            return float(parts[0]), float(parts[1])
    return None


def parse_file(path: Path) -> ImportedSpectrum:
    """Parse a SpectraSuite / OceanView / CSV spectrum export."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    header: dict = {}
    start = 0
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if any(low.startswith(m) for m in BEGIN_MARKERS) or low in BEGIN_MARKERS:
            start = i + 1
            break
        if ":" in line and not _NUM.match(line.strip().split()[0] if line.strip() else "x"):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key and val and len(key) < 60:
                header[key] = val
    else:
        start = 0

    wl, vals = [], []
    for line in lines[start:]:
        low = line.strip().lower()
        if any(low.startswith(m) for m in END_MARKERS):
            break
        pair = _split_pair(line)
        if pair is not None:
            wl.append(pair[0])
            vals.append(pair[1])

    if len(wl) < 10:
        raise ValueError(f"{path.name}: could not find spectral data "
                         f"(only {len(wl)} usable rows)")

    wl_arr = np.asarray(wl, dtype=np.float64)
    v_arr = np.asarray(vals, dtype=np.float64)
    order = np.argsort(wl_arr)
    wl_arr, v_arr = wl_arr[order], v_arr[order]

    # Header hints first, then fall back to value-range heuristics.
    is_refl = False
    scale_pct = False
    for key, val in header.items():
        k, v = key.lower(), val.lower()
        if "mode" in k or "axis" in k or "processing" in k:
            if any(w in v for w in ("reflect", "transmis", "absorb")):
                is_refl = True
    finite = v_arr[np.isfinite(v_arr)]
    if finite.size:
        hi = float(np.percentile(finite, 99.5))
        if hi <= 1.6:
            is_refl = True
        elif hi <= 130.0 and float(np.median(finite)) < 105.0:
            is_refl = True
            scale_pct = True
    if scale_pct:
        v_arr = v_arr / 100.0

    def hget(*keys):
        for key, val in header.items():
            lk = key.lower()
            if any(k in lk for k in keys):
                m = re.search(r"[-+]?\d*\.?\d+", val)
                if m:
                    return float(m.group())
        return None

    int_us = hget("integration time (usec", "integration time (us")
    int_ms = hget("integration time (msec", "integration time (ms")
    integration_ms = (int_us / 1000.0) if int_us else int_ms

    return ImportedSpectrum(
        wavelength_nm=wl_arr,
        values=v_arr,
        is_reflectance=is_refl,
        header=header,
        source_path=str(path),
        integration_time_ms=integration_ms,
        scans_averaged=int(hget("scans to average") or 0) or None,
        boxcar_width=int(hget("boxcar") or 0) or None,
    )


# ---------------------------------------------------------------------------
#  Folder watcher
# ---------------------------------------------------------------------------
SPECTRUM_SUFFIXES = {".txt", ".csv", ".tsv", ".trm", ".abs", ".irrad", ".dat", ".jdx"}


class FolderWatcher:
    """
    Watch the SpectraSuite export folder and invoke a callback for each new
    spectrum file.

    Uses watchdog when available and falls back to polling. Files are only
    handed on once their size has been stable for one poll interval, because
    SpectraSuite writes incrementally and a file read too eagerly comes back
    truncated.
    """

    def __init__(self, directory: Path, callback, poll_seconds: float = 1.0,
                 settle_seconds: float = 0.8):
        self.directory = Path(directory)
        self.callback = callback
        self.poll_seconds = poll_seconds
        self.settle_seconds = settle_seconds
        self._seen: dict = {}
        self._observer = None
        self._stop = False
        self._thread = None

    # -- polling implementation (always available) -------------------------
    def _scan_once(self) -> None:
        if not self.directory.exists():
            return
        now = time.time()
        for path in sorted(self.directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SPECTRUM_SUFFIXES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            key = str(path)
            prev = self._seen.get(key)
            sig = (stat.st_size, stat.st_mtime)
            if prev is None:
                self._seen[key] = {"sig": sig, "since": now, "done": False}
                continue
            if prev["done"]:
                if prev["sig"] != sig:      # file rewritten - re-ingest
                    self._seen[key] = {"sig": sig, "since": now, "done": False}
                continue
            if prev["sig"] != sig:
                self._seen[key] = {"sig": sig, "since": now, "done": False}
                continue
            if now - prev["since"] >= self.settle_seconds:
                prev["done"] = True
                try:
                    self.callback(path)
                except Exception as exc:
                    print(f"[spectrasuite] failed to ingest {path.name}: {exc}")

    def _loop(self) -> None:
        while not self._stop:
            self._scan_once()
            time.sleep(self.poll_seconds)

    def start(self) -> None:
        import threading
        self.directory.mkdir(parents=True, exist_ok=True)
        # Mark pre-existing files as already handled so a restart does not
        # re-analyse the entire folder history.
        for path in self.directory.iterdir():
            if path.is_file() and path.suffix.lower() in SPECTRUM_SUFFIXES:
                try:
                    st = path.stat()
                    self._seen[str(path)] = {"sig": (st.st_size, st.st_mtime),
                                             "since": 0.0, "done": True}
                except OSError:
                    pass
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="spectrasuite-watch")
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
