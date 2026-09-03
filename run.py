"""
Launch the AI Spectroscopic Model dashboard.

    python run.py                 # local only, no password
    python run.py --tunnel        # also publish a public HTTPS URL
    python run.py --port 8080
    python run.py --host 0.0.0.0  # reachable from the LAN

About --tunnel
--------------
The spectrometer is a USB device on this machine, so the measurement has to
happen here. A cloud host cannot read it - a container in a datacentre has no
USB bus. What a tunnel does is publish *this* running instance at a public
HTTPS address, so the console works from anywhere while the hardware stays
plugged in here. That is the only arrangement in which a public URL reads the
real instrument.

Publishing also means the API becomes reachable by anyone with the link, and
that API can start acquisitions, overwrite references and read every stored
analysis. So --tunnel generates a password and turns on authentication; there
is no way to publish without it.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import auth as authmod          # noqa: E402
from backend import config as cfg_mod        # noqa: E402

ROOT = Path(__file__).resolve().parent
CLOUDFLARED = ROOT / "tools" / "cloudflared.exe"
URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


def _redirect_output_if_headless() -> None:
    """
    Give the process somewhere to write when it has no console.

    pythonw.exe - the interpreter the scheduled task uses so no black window
    appears - leaves sys.stdout and sys.stderr as None. The first print() then
    raises AttributeError and the process dies before it has done anything, with
    nothing on screen and only "exit code 1" in Task Scheduler to go on. Uvicorn's
    logging would hit the same wall.

    So when there is no stream, send both to a log file instead. That also means
    an unattended console leaves a trace worth reading when something goes wrong.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_dir = ROOT / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = open(log_dir / "console.log", "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    print(f"\n{'=' * 72}\n  started headless {time.strftime('%Y-%m-%d %H:%M:%S')}")


_redirect_output_if_headless()


def wait_for_server(port: int, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def find_cloudflared() -> str | None:
    if CLOUDFLARED.exists():
        return str(CLOUDFLARED)
    from shutil import which
    return which("cloudflared")


def start_tunnel(port: int, password: str, user: str) -> subprocess.Popen | None:
    exe = find_cloudflared()
    if not exe:
        print("\n  cloudflared is not installed. Get it with:")
        print("      python scripts/fetch_cloudflared.py")
        print("  or   winget install --id Cloudflare.cloudflared\n")
        return None

    if not wait_for_server(port):
        print("\n  The local server did not come up; not starting the tunnel.\n")
        return None

    proc = subprocess.Popen(
        [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    def reader():
        url = None
        for line in proc.stdout:
            if url is None:
                m = URL_RE.search(line)
                if m:
                    url = m.group(0)
                    banner(url, password, user)
            # cloudflared is chatty; surface only what matters after the URL.
            low = line.lower()
            if "err" in low and "error=nil" not in low:
                print("  [tunnel] " + line.rstrip())

    threading.Thread(target=reader, daemon=True, name="tunnel-reader").start()
    return proc


def banner(url: str, password: str, user: str) -> None:
    line = "=" * 72
    print(f"\n{line}")
    print("  PUBLIC URL IS LIVE")
    print(line)
    print(f"\n    {url}\n")
    print(f"    username : {user}")
    print(f"    password : {password}\n")
    print("  This address reaches the spectrometer plugged into THIS machine,")
    print("  so leave this window open. Closing it takes the URL down.")
    print("  The address is temporary and changes each time you restart.")
    print(f"{line}\n")


def main():
    cfg = cfg_mod.load_config()
    ap = argparse.ArgumentParser(description="AI Spectroscopic Model server")
    ap.add_argument("--host", default=cfg.get_path("server.host", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(cfg.get_path("server.port", 8000)))
    ap.add_argument("--tunnel", action="store_true",
                    help="publish a public HTTPS URL via Cloudflare (adds a password)")
    ap.add_argument("--password", default=None,
                    help="use this password instead of a generated one")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    cfg_mod.ensure_dirs()

    password = None
    if args.tunnel:
        password = args.password or os.environ.get(authmod.ENV_PASSWORD) \
            or authmod.generate_password()
        os.environ[authmod.ENV_PASSWORD] = password
    elif args.password:
        password = args.password
        os.environ[authmod.ENV_PASSWORD] = password

    local = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    print("=" * 72)
    print("  AI Spectroscopic Model  -  Ocean Optics USB4000")
    print("=" * 72)
    print(f"  Local     : {local}")
    print(f"  API docs  : {local}docs")
    print(f"  Watching  : {cfg.resolve('spectrasuite.watch_dir')}")
    # Report what is actually in force, not just what this invocation set.
    # Credentials also live in data/credentials.json so unattended starts keep
    # the same password, and a banner that ignored that would claim the console
    # is open when it is not.
    effective = password or authmod.password()
    if effective:
        print(f"  Password  : {effective}   (user: {authmod.username()})")
    else:
        print("  Password  : none - open to anyone who can reach this port")
    if args.tunnel:
        print("  Tunnel    : starting, the public URL appears below in a few seconds")
    print("=" * 72)

    tunnel = None
    if args.tunnel:
        threading.Thread(
            target=lambda: globals().__setitem__(
                "_tunnel", start_tunnel(args.port, password, authmod.username())),
            daemon=True, name="tunnel-start").start()
    elif not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.6), webbrowser.open(local)),
                         daemon=True).start()

    import uvicorn
    try:
        uvicorn.run("backend.main:app", host=args.host, port=args.port,
                    reload=args.reload, log_level="info")
    finally:
        tunnel = globals().get("_tunnel")
        if tunnel is not None:
            print("\n  stopping tunnel...")
            try:
                tunnel.send_signal(signal.SIGTERM)
                tunnel.wait(timeout=8)
            except Exception:
                tunnel.kill()


if __name__ == "__main__":
    main()
