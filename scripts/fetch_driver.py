"""
Download the Ocean Optics WinUSB driver package used by INSTALL-DRIVER.ps1.

    python scripts/fetch_driver.py

The package ships with python-seabreeze. It contains signed INF files that bind
Microsoft's in-box WinUSB driver to each Ocean Optics product ID; OOI_USB4000.inf
matches USB\VID_2457&PID_1022. Downloading it here means the elevated installer
does not need network access.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://raw.githubusercontent.com/ap--/python-seabreeze/master/"
       "os_support/windows-driver-files.zip")
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "drivers"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    try:
        data = urllib.request.urlopen(URL, timeout=90).read()
    except Exception as exc:
        print(f"  failed: {exc}")
        print("  Download it manually and unzip into drivers/win/")
        return 1
    print(f"  {len(data) / 1024:.0f} KB")

    (DEST / "windows-driver-files.zip").write_bytes(data)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(DEST / "win")
        n = len(z.namelist())

    inf = DEST / "win" / "OOI_USB4000.inf"
    print(f"  extracted {n} files to {DEST / 'win'}")
    print(f"  USB4000 INF present: {inf.exists()}")
    if inf.exists():
        print("\nNow run INSTALL-DRIVER.bat (it will ask for administrator rights).")
    return 0 if inf.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
