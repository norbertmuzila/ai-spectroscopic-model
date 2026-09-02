"""
Download the cloudflared binary used by `python run.py --tunnel`.

    python scripts/fetch_cloudflared.py

A single ~55 MB executable, no installer and no administrator rights. It goes
into tools/ where run.py looks for it.
"""
from __future__ import annotations

import platform
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
ASSETS = {
    ("Windows", "AMD64"): ("cloudflared-windows-amd64.exe", "cloudflared.exe"),
    ("Windows", "ARM64"): ("cloudflared-windows-arm64.exe", "cloudflared.exe"),
    ("Linux", "x86_64"): ("cloudflared-linux-amd64", "cloudflared"),
    ("Linux", "aarch64"): ("cloudflared-linux-arm64", "cloudflared"),
    ("Darwin", "arm64"): ("cloudflared-darwin-arm64.tgz", "cloudflared.tgz"),
    ("Darwin", "x86_64"): ("cloudflared-darwin-amd64.tgz", "cloudflared.tgz"),
}


def main() -> int:
    key = (platform.system(), platform.machine())
    if key not in ASSETS:
        print(f"No cloudflared build listed for {key}.")
        print("Download it manually from https://github.com/cloudflare/cloudflared/releases")
        return 1

    asset, local = ASSETS[key]
    dest = ROOT / "tools" / local
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = BASE + asset
    print(f"downloading {url}")
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "spectral-console"})
        with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:
        print(f"  failed: {exc}")
        return 1

    size = dest.stat().st_size / 1e6
    print(f"  {size:.1f} MB in {time.time() - t0:.0f}s -> {dest}")
    if platform.system() != "Windows":
        dest.chmod(0o755)
    print("\nNow run:  python run.py --tunnel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
