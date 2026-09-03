"""
Give the console a permanent public HTTPS address using Tailscale Funnel.

    python scripts/setup_funnel.py

Why Funnel rather than a quick tunnel
-------------------------------------
A quick tunnel hands out a fresh random hostname every time it starts, so the
address you shared yesterday is dead today. Funnel publishes the machine under a
stable name derived from your tailnet - something like
``spectral-console.tail1a2b.ts.net`` - which survives reboots, restarts and
network changes, with a real certificate and no domain to buy.

The traffic still terminates on this machine. That is the point: the
spectrometer is plugged in here, so this is the only arrangement where a public
address reads real hardware rather than a simulator.

Funnel is public. Anyone who knows the hostname can reach it, so the console's
password is mandatory here and this script refuses to run without one.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import auth as authmod          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URL_FILE = ROOT / "data" / "public_url.txt"
CANDIDATES = [
    r"C:\Program Files\Tailscale\tailscale.exe",
    r"C:\Program Files (x86)\Tailscale\tailscale.exe",
]


def tailscale() -> str | None:
    for c in CANDIDATES:
        if Path(c).exists():
            return c
    return shutil.which("tailscale")


def run(exe: str, *args, timeout: float = 90.0):
    return subprocess.run([exe, *args], capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace")


def status(exe: str) -> dict:
    r = run(exe, "status", "--json", timeout=30)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def main(port: int = 8000) -> int:
    print("=" * 74)
    print("  Permanent public address via Tailscale Funnel")
    print("=" * 74)

    exe = tailscale()
    if not exe:
        print("\n  Tailscale is not installed.")
        print("  Install it with:  winget install --id Tailscale.Tailscale")
        return 1
    print(f"\n  tailscale: {exe}")

    st = status(exe)
    state = st.get("BackendState")
    if state != "Running":
        print(f"\n  Tailscale is not signed in (state: {state}).")
        r = run(exe, "status", timeout=20)
        for line in (r.stdout + r.stderr).splitlines():
            if "login.tailscale.com" in line:
                print(f"  Sign in here, then run this again:\n    {line.strip()}")
                break
        else:
            print("  Run:  tailscale up --hostname=spectral-console")
        return 1

    dns = (st.get("Self") or {}).get("DNSName", "").rstrip(".")
    if not dns:
        print("\n  Signed in, but this machine has no DNS name yet. Wait a few "
              "seconds and run again.")
        return 1
    print(f"  machine  : {dns}")

    # A public address without a password is not something to hand out.
    pw = authmod.password()
    if not pw:
        pw = authmod.generate_password()
        authmod.save_password(pw)
        print(f"\n  No password was set, so one was generated: {pw}")
    else:
        print(f"  password : {pw}  (user: {authmod.username()})")

    print(f"\n  Publishing http://127.0.0.1:{port} ...")
    r = run(exe, "funnel", "--bg", str(port))
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if line.strip():
            print("    " + line.strip())

    if r.returncode != 0:
        low = out.lower()
        if "funnel" in low and ("attribute" in low or "enable" in low or "permission" in low):
            print("\n  Funnel is not enabled for your tailnet yet. Tailscale prints a")
            print("  one-click link above - open it, approve, then run this again.")
            print("  Otherwise enable it at https://login.tailscale.com/admin/acls")
        return 1

    url = f"https://{dns}"
    URL_FILE.parent.mkdir(parents=True, exist_ok=True)
    URL_FILE.write_text(url, encoding="utf-8")

    print("\n  Checking it from the public internet...")
    ok = False
    for attempt in range(12):
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=10) as resp:
                if resp.status == 200:
                    ok = True
                    break
        except Exception:
            time.sleep(5)

    line = "=" * 74
    print(f"\n{line}")
    if ok:
        print("  PERMANENT ADDRESS IS LIVE")
    else:
        print("  ADDRESS REGISTERED (not answering yet)")
    print(line)
    print(f"\n    {url}\n")
    print(f"    username : {authmod.username()}")
    print(f"    password : {pw}\n")
    if ok:
        print("  This address does not change. It works after a reboot, on any")
        print("  network, from any device.")
    else:
        print("  The certificate can take a minute on first use, and the console")
        print("  itself must be running. Start it and try the URL in a browser.")
    print(line)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
