"""
Launch the AI Spectroscopic Model dashboard.

    python run.py                 # serve on 127.0.0.1:8000 and open a browser
    python run.py --port 8080
    python run.py --host 0.0.0.0  # reachable from other machines on the LAN
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import config as cfg_mod


def main():
    cfg = cfg_mod.load_config()
    ap = argparse.ArgumentParser(description="AI Spectroscopic Model server")
    ap.add_argument("--host", default=cfg.get_path("server.host", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(cfg.get_path("server.port", 8000)))
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = ap.parse_args()

    cfg_mod.ensure_dirs()

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    print("=" * 70)
    print("  AI Spectroscopic Model  -  Ocean Optics USB4000")
    print("=" * 70)
    print(f"  Dashboard : {url}")
    print(f"  API docs  : {url}docs")
    print(f"  Watching  : {cfg.resolve('spectrasuite.watch_dir')}")
    print("=" * 70)

    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(1.6), webbrowser.open(url)),
            daemon=True).start()

    import uvicorn
    uvicorn.run("backend.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
