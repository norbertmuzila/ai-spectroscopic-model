"""
Measure the wavelength range this bench can actually work over, and write it
into the config.

    python scripts/calibrate_range.py            # measure and report
    python scripts/calibrate_range.py --write    # also update config.yaml

Run it with the lamp on and a white reflectance standard (Spectralon or PTFE)
under the probe, exactly as you would before taking a white reference.

Why this is worth doing
-----------------------
The detector's span and the *usable* span are different numbers, and only the
second one matters. A USB4000 reports its full grating range - on this unit
195-913 nm - but at both extremes the lamp output, grating efficiency and
silicon response all fall away together. Where the white reference approaches
the dark level, reflectance becomes one small number divided by another, and
continuum removal turns that noise into absorption bands that look entirely
convincing.

So the honest lower bound is set by your light source, not the detector: a
tungsten-halogen lamp is dark below roughly 340 nm, a deuterium source reaches
much further into the UV. This script measures where your signal actually lives
and writes that range into the config, which is what the engine uses to decide
which mineral diagnostics it is allowed to claim.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config as cfg_mod                  # noqa: E402
from backend.hardware.device import open_hardware      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="update config/config.yaml")
    ap.add_argument("--integration-ms", type=float, default=100.0)
    ap.add_argument("--scans", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="fraction of peak white signal that counts as usable")
    args = ap.parse_args()

    print("=" * 74)
    print("  Usable wavelength range")
    print("=" * 74)

    try:
        dev = open_hardware()
    except Exception as exc:
        print(f"\n  Cannot open the spectrometer: {exc}")
        print("  Run  python scripts/diagnose.py  first.")
        return 1

    wl = dev.wavelengths()
    print(f"\n  {dev.model}  serial {dev.serial}")
    print(f"  detector span : {wl.min():.2f} - {wl.max():.2f} nm, {wl.size} pixels")

    dev.set_integration_time(args.integration_ms)
    dev.set_averaging(args.scans)

    input("\n  Block the light path (cap the probe), then press Enter for DARK... ")
    dark = dev.acquire()
    print(f"    dark level: {np.median(dark):.0f} counts")

    input("  Now illuminate the white standard, then press Enter for WHITE... ")
    white = dev.acquire()
    print(f"    white peak: {white.max():.0f} counts at {wl[white.argmax()]:.1f} nm")

    if white.max() >= 65000:
        print("\n  WARNING: the white reference is saturating. Lower the integration"
              "\n  time and run this again, or the range will read too wide.")

    signal = white - dark
    peak = float(np.percentile(signal, 99.5))
    if peak <= 0:
        print("\n  The white reference is no brighter than the dark. Is the lamp on"
              "\n  and the fibre connected?")
        dev.close()
        return 1

    good = signal >= args.threshold * peak
    if good.sum() < 20:
        print("\n  Almost no channel carries signal. Check the lamp and the fibre.")
        dev.close()
        return 1

    lo, hi = float(wl[good].min()), float(wl[good].max())
    print(f"\n  Signal above {args.threshold * 100:.0f}% of peak:  "
          f"{lo:.1f} - {hi:.1f} nm   ({good.sum()} pixels)")

    print("\n  Signal by region (white minus dark):")
    for a, b in [(190, 300), (300, 340), (340, 400), (400, 500), (500, 700),
                 (700, 850), (850, 913)]:
        m = (wl >= a) & (wl < b)
        if m.sum():
            frac = signal[m].max() / peak
            mark = "usable" if frac >= args.threshold else "too dark"
            print(f"    {a:4d}-{b:4d} nm : peak {signal[m].max():8.0f} "
                  f"({frac * 100:5.1f}% of max)  {mark}")

    dev.close()

    lo_r, hi_r = float(np.ceil(lo)), float(np.floor(hi))
    print(f"\n  Recommended config range: {lo_r:.0f} - {hi_r:.0f} nm")

    if args.write:
        path = cfg_mod.CONFIG_PATH
        text = path.read_text(encoding="utf-8")
        import re
        text = re.sub(r"wavelength_min_nm:\s*[\d.]+", f"wavelength_min_nm: {lo_r:.1f}", text)
        text = re.sub(r"wavelength_max_nm:\s*[\d.]+", f"wavelength_max_nm: {hi_r:.1f}", text)
        text = re.sub(r"grid_min_nm:\s*[\d.]+", f"grid_min_nm: {lo_r:.1f}", text)
        text = re.sub(r"grid_max_nm:\s*[\d.]+", f"grid_max_nm: {hi_r:.1f}", text)
        path.write_text(text, encoding="utf-8")
        print(f"  written to {path}")
        print("\n  The model is tied to the wavelength grid, so retrain now:")
        print("      python scripts/train.py")
    else:
        print("  Re-run with --write to save it, then retrain:  python scripts/train.py")

    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
