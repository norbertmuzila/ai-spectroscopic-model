"""
Spectrometer connection diagnostic.

    python scripts/diagnose.py

Run this whenever the dashboard will not read the instrument. It reports what is
actually wrong - library, USB backend, cable, driver, or another program holding
the device - and the specific step that fixes it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.hardware import diagnostics  # noqa: E402

BAR = "=" * 74
MARK = {"ok": "[ OK ]", "warning": "[WARN]", "error": "[FAIL]"}


def main() -> int:
    print(BAR)
    print("  USB4000 connection diagnostic")
    print(BAR)

    d = diagnostics.diagnose()

    print(f"\nPython {d['python']} on {d['platform']}")

    lib = d["library"]
    print(f"\n{'-' * 74}\nLibrary\n{'-' * 74}")
    if lib["seabreeze_installed"]:
        print(f"  seabreeze {lib['version']} installed")
        for name, info in lib["backends"].items():
            if info["available"]:
                print(f"    {name:<14} available")
            else:
                print(f"    {name:<14} UNAVAILABLE - {info.get('error', '')[:80]}")
    else:
        print(f"  seabreeze NOT installed - {lib.get('error', '')}")

    ub = d["usb_backend"]
    print(f"\n{'-' * 74}\nUSB backend\n{'-' * 74}")
    print(f"  pyusb            {'yes' if ub.get('pyusb') else 'NO'}")
    print(f"  libusb backend   {'yes (' + str(ub.get('libusb_source', '')) + ')' if ub.get('libusb') else 'NO'}")
    if ub.get("n_usb_devices") is not None:
        print(f"  USB devices seen {ub['n_usb_devices']}")

    bus = d["bus_scan"]
    print(f"\n{'-' * 74}\nOcean Optics devices on the bus\n{'-' * 74}")
    if bus.get("devices"):
        for dev in bus["devices"]:
            print(f"  {dev['model']} ({dev['product_id']}) bus {dev['bus']} addr {dev['address']}")
    else:
        print("  none")

    pnp = d["windows_pnp"]
    if pnp.get("checked"):
        print(f"\n{'-' * 74}\nWhat Windows sees\n{'-' * 74}")
        if pnp["present"]:
            for p in pnp["present"]:
                drv = p["driver_service"] or "NO DRIVER BOUND"
                print(f"  ATTACHED  {p['name']}  [{p['status']}]  driver: {drv}")
        else:
            print("  ATTACHED  none")
        if pnp["ghost"]:
            n = len(pnp["ghost"])
            print(f"  remembered but not plugged in: {n} "
                  f"{'entries' if n != 1 else 'entry'}")
            for g in pnp["ghost"]:
                print(f"      {g['name']}  ({g['instance']})")
            print("      ^ these are historical records, not live devices. Device Manager")
            print("        shows them the same way, which is why they mislead.")

    op = d["open_attempt"]
    print(f"\n{'-' * 74}\nseabreeze open attempt\n{'-' * 74}")
    if op.get("opened"):
        o = op["opened"]
        print(f"  SUCCESS  {o['model']} serial {o['serial']}")
        print(f"           {o['pixels']} pixels, {o['wavelength_min_nm']}-"
              f"{o['wavelength_max_nm']} nm, peak {o['peak_counts']} counts")
    else:
        print(f"  devices listed: {len(op.get('listed', []))}")
        if op.get("error"):
            print(f"  error: {op['error']}")

    print(f"\n{BAR}")
    print(f"  {MARK.get(d['severity'], '[????]')}  {d['verdict']}")
    print(BAR)
    for i, step in enumerate(d["next_steps"], 1):
        print(f"\n  {i}. {step}")
    print()

    if "--json" in sys.argv:
        print(json.dumps(d, indent=2, default=str))

    return 0 if d["hardware_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
