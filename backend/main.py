"""
FastAPI backend: device control, live streaming, analysis, reporting.

Run with::

    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

or simply ``python run.py``. The dashboard is served from ``/``.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import auth as authmod
from backend import config as cfg_mod
from backend.db import Database
from backend.hardware import device as devmod
from backend.hardware import diagnostics as diagmod
from backend.hardware import spectrasuite as ssmod
from backend.library import minerals as mindb
from backend.models.engine import get_engine
from backend.reporting import report as repmod

cfg_mod.ensure_dirs()
CFG = cfg_mod.load_config()
ROOT = cfg_mod.ROOT

app = FastAPI(title="AI Spectroscopic Model", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# Inactive unless SPECTRO_PASSWORD is set, which run.py --tunnel does. Registered
# after CORS so it runs first: an unauthenticated request should be rejected
# before anything else looks at it.
app.middleware("http")(authmod.middleware)


@app.get("/api/health")
def health():
    """Unauthenticated liveness probe, so a tunnel or host can check the app."""
    return {"ok": True, "service": "spectral-console"}

DB = Database(CFG.resolve("server.database"))
REPORTS_DIR = CFG.resolve("server.reports_dir")


# ---------------------------------------------------------------------------
#  Shared state
# ---------------------------------------------------------------------------
class State:
    def __init__(self):
        self.device: devmod.BaseSpectrometer | None = None
        self.engine = None
        self.session_id: int | None = None
        self.watcher: ssmod.FolderWatcher | None = None
        self.streaming = False
        self.clients: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last_analysis: dict | None = None

    def require_device(self):
        """
        Return the connected device, or refuse.

        Every acquisition path goes through here so that no endpoint can
        conjure a simulator on demand. Opening one implicitly would mean a
        measurement request answered with synthetic data that looks exactly
        like a real reading.
        """
        if self.device is None:
            raise HTTPException(409, "No spectrometer connected. Press Connect on the "
                                     "dashboard first (or tick Simulator mode).")
        return self.device

    def ensure_engine(self):
        if self.engine is None:
            self.engine = get_engine()
        return self.engine


ST = State()


async def broadcast(message: dict) -> None:
    if not ST.clients:
        return
    payload = json.dumps(message)
    dead = []
    for ws in list(ST.clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ST.clients.discard(ws)


def broadcast_threadsafe(message: dict) -> None:
    """Push to websocket clients from a non-async thread (the folder watcher)."""
    if ST.loop is None:
        return
    asyncio.run_coroutine_threadsafe(broadcast(message), ST.loop)


# ---------------------------------------------------------------------------
#  Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    ST.loop = asyncio.get_running_loop()
    ST.ensure_engine()

    # Demo mode, for a hosted deployment. A cloud container has no USB port, so
    # the simulator is connected up front with references already taken -
    # otherwise the page opens to a dead console and looks broken.
    if os.environ.get("SPECTRO_DEMO", "").lower() in ("1", "true", "yes"):
        try:
            dev = devmod.open_device(prefer_simulator=True)
            dev.set_integration_time(95)
            dev.set_averaging(12)
            dev.take_reference("dark")
            dev.take_reference("white")
            dev.load_sample("Hematite")
            ST.device = dev
            print("[startup] demo mode: simulator connected, references taken")
        except Exception as exc:
            print(f"[startup] demo mode failed: {exc}")
    if CFG.get_path("spectrasuite.watch_enabled", True):
        watch_dir = CFG.resolve("spectrasuite.watch_dir")
        ST.watcher = ssmod.FolderWatcher(watch_dir, _on_spectrasuite_file)
        ST.watcher.start()
        print(f"[startup] watching SpectraSuite exports in {watch_dir}")


@app.on_event("shutdown")
async def shutdown():
    if ST.watcher:
        ST.watcher.stop()
    if ST.device:
        ST.device.close()


def _on_spectrasuite_file(path: Path) -> None:
    """Called by the folder watcher for each new SpectraSuite export."""
    try:
        imported = ssmod.parse_file(path)
        engine = ST.ensure_engine()
        result = engine.analyze(
            wavelength_nm=imported.wavelength_nm,
            values=imported.values,
            already_reflectance=imported.is_reflectance,
            sample_label=path.stem,
            metadata={"source": "spectrasuite", **imported.as_dict()},
        ).as_dict()
        DB.save_analysis(result, ST.session_id, source="spectrasuite")
        ST.last_analysis = result
        broadcast_threadsafe({"type": "analysis", "data": result})
        print(f"[spectrasuite] analysed {path.name} -> "
              f"{result['identification'].get('mineral')} "
              f"({result['identification'].get('confidence')})")
    except Exception as exc:
        broadcast_threadsafe({"type": "error",
                              "message": f"{path.name}: {exc}"})


# ---------------------------------------------------------------------------
#  Models
# ---------------------------------------------------------------------------
class DeviceSettings(BaseModel):
    integration_time_ms: float | None = None
    scans_to_average: int | None = None
    boxcar_width: int | None = None


class ConnectRequest(BaseModel):
    simulator: bool = False


class MeasureRequest(BaseModel):
    sample_label: str = "Unlabelled sample"
    scans: int | None = None
    session_id: int | None = None
    notes: str = ""


class SessionRequest(BaseModel):
    name: str
    operator: str = ""
    site: str = ""
    notes: str = ""


class SimulatorSample(BaseModel):
    mixture: dict
    grain_size: float = 1.0
    weathering: float = 0.0


# ---------------------------------------------------------------------------
#  Status
# ---------------------------------------------------------------------------
@app.get("/api/status")
def status():
    engine = ST.ensure_engine()
    dev = ST.device
    return {
        "device": dev.info() if dev else {"connected": False},
        "connected": dev is not None,
        "streaming": ST.streaming,
        "session_id": ST.session_id,
        "engine": {
            "model_loaded": engine.model_loaded,
            "library": engine.library.summary(),
            "instrument_range_nm": list(engine.instrument_range),
            "validation": engine.model_metrics.get("test", {}),
            "temperature": engine.calibration.scaler.temperature,
            "conformal_alpha": engine.calibration.conformal.alpha,
        },
        "spectrasuite": {
            "watching": ST.watcher is not None,
            "watch_dir": str(CFG.resolve("spectrasuite.watch_dir")),
            "exclusive_device_mode": bool(
                CFG.get_path("spectrasuite.exclusive_device_mode", False)),
        },
        "stats": DB.stats(),
    }


@app.get("/api/minerals")
def minerals():
    engine = ST.ensure_engine()
    lo, hi = engine.instrument_range
    out = []
    for m in mindb.ALL_MINERALS:
        cov = engine.matcher.diagnostic_coverage(m.name, lo, hi)
        out.append({
            "name": m.name,
            "group": m.group,
            "formula": m.formula,
            "environments": list(m.env),
            "diagnostic_bands_nm": [round(c * 1000.0) for c in m.diagnostic_centres],
            "diagnostic_coverage": round(cov, 2),
            "in_library": m.name in engine.library.classes,
            "notes": m.notes,
        })
    out.sort(key=lambda d: (-d["diagnostic_coverage"], d["group"], d["name"]))
    return {"minerals": out, "summary": mindb.summary(),
            "instrument_range_nm": [lo, hi]}


# ---------------------------------------------------------------------------
#  Device
# ---------------------------------------------------------------------------
@app.post("/api/device/connect")
def device_connect(req: ConnectRequest):
    """
    Open the spectrometer.

    When ``simulator`` is false and no instrument is reachable this returns a
    failure carrying the full diagnosis, and leaves the device disconnected. It
    deliberately does not quietly hand back a simulator: an operator watching
    synthetic spectra scroll past, believing they are looking at their sample,
    is far worse off than one told plainly that the cable is not connected.
    """
    if ST.device is not None:
        ST.device.close()
        ST.device = None

    if not req.simulator:
        try:
            dev = devmod.open_device(prefer_simulator=False, allow_fallback=False)
        except Exception as exc:
            diag = diagmod.diagnose()
            return {
                "ok": False,
                "connected": False,
                "reason": str(exc),
                "diagnostics": diag,
                "hint": "Fix the issue above, or tick Simulator mode to explore "
                        "the system without hardware.",
            }
        ST.device = dev
    else:
        ST.device = devmod.open_device(prefer_simulator=True)

    dev = ST.device
    dev.set_integration_time(float(CFG.get_path("instrument.default_integration_time_ms", 100)))
    dev.set_averaging(int(CFG.get_path("instrument.default_scans_to_average", 10)))
    dev.set_boxcar(int(CFG.get_path("instrument.default_boxcar_width", 2)))
    return {"ok": True, "connected": True, "device": dev.info(),
            "available_hardware": devmod.list_available()}


@app.get("/api/diagnostics")
def diagnostics():
    """Full hardware diagnosis: library, USB backend, bus, driver, open attempt."""
    return diagmod.diagnose()


@app.post("/api/device/disconnect")
def device_disconnect():
    ST.streaming = False
    if ST.device:
        ST.device.close()
        ST.device = None
    return {"ok": True}


@app.post("/api/device/settings")
def device_settings(s: DeviceSettings):
    dev = ST.require_device()
    if s.integration_time_ms is not None:
        dev.set_integration_time(s.integration_time_ms)
    if s.scans_to_average is not None:
        dev.set_averaging(s.scans_to_average)
    if s.boxcar_width is not None:
        dev.set_boxcar(s.boxcar_width)
    return {"ok": True, "device": dev.info()}


@app.post("/api/device/reference/{kind}")
def device_reference(kind: str):
    if kind not in ("dark", "white"):
        raise HTTPException(400, "kind must be 'dark' or 'white'")
    dev = ST.require_device()
    ref = dev.take_reference(kind)
    return {"ok": True, "kind": kind,
            "integration_time_ms": ref.integration_time_ms,
            "peak_counts": float(np.max(ref.counts)),
            "device": dev.info()}


@app.delete("/api/device/reference/{kind}")
def device_clear_reference(kind: str):
    dev = ST.require_device()
    dev.clear_reference(kind)
    return {"ok": True, "device": dev.info()}


@app.get("/api/device/preview")
def device_preview():
    """One raw frame, for the live plot before a full analysis is run."""
    dev = ST.require_device()
    counts = dev.acquire(scans=1)
    wl = dev.wavelengths()
    stride = max(1, wl.size // 900)
    return {
        "wavelength_nm": [round(float(v), 2) for v in wl[::stride]],
        "counts": [round(float(v), 1) for v in counts[::stride]],
        "peak_counts": float(counts.max()),
        "saturated": bool(counts.max() >= 65000),
        "integration_time_ms": dev.integration_time_ms,
    }


@app.post("/api/simulator/sample")
def simulator_sample(req: SimulatorSample):
    dev = ST.require_device()
    if not getattr(dev, "is_simulated", False):
        raise HTTPException(400, "connected device is real hardware, not the simulator")
    dev.load_mixture(req.mixture, grain_size=req.grain_size, weathering=req.weathering)
    return {"ok": True, "loaded": dev.loaded_sample}


# ---------------------------------------------------------------------------
#  Measurement + analysis
# ---------------------------------------------------------------------------
def _run_analysis(dev, sample_label: str, scans: int | None, session_id: int | None,
                  notes: str = "") -> dict:
    engine = ST.ensure_engine()
    counts = dev.acquire(scans)
    result = engine.analyze(
        wavelength_nm=dev.wavelengths(),
        values=counts,
        dark=dev.dark.counts if dev.dark else None,
        white=dev.white.counts if dev.white else None,
        nonlinearity_coeffs=(dev.nonlinearity_coeffs
                             if CFG.get_path("instrument.nonlinearity_correction", True)
                             else None),
        sample_label=sample_label,
        reference_age_minutes=dev.reference_age_minutes(),
        metadata={
            "source": "live",
            "notes": notes,
            "device": dev.info(),
            "ground_truth": getattr(dev, "loaded_sample", None) if dev.is_simulated else None,
        },
    ).as_dict()
    DB.save_analysis(result, session_id or ST.session_id, source="live")
    ST.last_analysis = result
    return result


@app.post("/api/measure")
async def measure(req: MeasureRequest):
    dev = ST.require_device()
    result = await asyncio.to_thread(_run_analysis, dev, req.sample_label,
                                     req.scans, req.session_id, req.notes)
    await broadcast({"type": "analysis", "data": result})
    return result


@app.post("/api/import")
async def import_spectrum(file: UploadFile = File(...),
                          sample_label: str = ""):
    """Import a SpectraSuite / OceanView / CSV export and analyse it."""
    incoming = CFG.resolve("spectrasuite.watch_dir")
    incoming.mkdir(parents=True, exist_ok=True)
    dest = incoming / f"upload_{int(time.time())}_{Path(file.filename).name}"
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        imported = ssmod.parse_file(dest)
    except Exception as exc:
        raise HTTPException(400, f"could not parse {file.filename}: {exc}")

    engine = ST.ensure_engine()
    result = await asyncio.to_thread(
        lambda: engine.analyze(
            wavelength_nm=imported.wavelength_nm,
            values=imported.values,
            already_reflectance=imported.is_reflectance,
            sample_label=sample_label or Path(file.filename).stem,
            metadata={"source": "import", **imported.as_dict()},
        ).as_dict())
    DB.save_analysis(result, ST.session_id, source="import")
    ST.last_analysis = result
    await broadcast({"type": "analysis", "data": result})
    return result


# ---------------------------------------------------------------------------
#  Sessions and history
# ---------------------------------------------------------------------------
@app.post("/api/session")
def create_session(req: SessionRequest):
    sid = DB.create_session(req.name, req.operator, req.site, req.notes)
    ST.session_id = sid
    return {"ok": True, "session_id": sid, "session": DB.get_session(sid)}


@app.post("/api/session/{session_id}/end")
def end_session(session_id: int):
    DB.end_session(session_id)
    if ST.session_id == session_id:
        ST.session_id = None
    return {"ok": True}


@app.get("/api/sessions")
def list_sessions():
    return {"sessions": DB.list_sessions()}


@app.get("/api/analyses")
def list_analyses(session_id: int | None = None, limit: int = 100):
    return {"analyses": DB.list_analyses(session_id, limit)}


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str):
    result = DB.get_analysis(analysis_id)
    if result is None:
        raise HTTPException(404, "analysis not found")
    return result


@app.delete("/api/analyses/{analysis_id}")
def delete_analysis(analysis_id: str):
    return {"ok": DB.delete_analysis(analysis_id)}


# ---------------------------------------------------------------------------
#  Reports
# ---------------------------------------------------------------------------
@app.post("/api/report/{analysis_id}")
async def make_report(analysis_id: str):
    result = DB.get_analysis(analysis_id)
    if result is None:
        raise HTTPException(404, "analysis not found")
    paths = await asyncio.to_thread(repmod.generate, result, REPORTS_DIR)
    return {"ok": True, "analysis_id": analysis_id, "files": paths}


@app.get("/api/report/{analysis_id}/download")
async def download_report(analysis_id: str, fmt: str = "pdf"):
    if fmt not in ("pdf", "json", "csv"):
        raise HTTPException(400, "fmt must be pdf, json or csv")
    suffix = {"pdf": "_report.pdf", "json": "_data.json", "csv": "_spectrum.csv"}[fmt]
    path = REPORTS_DIR / f"{analysis_id}{suffix}"
    if not path.exists():
        result = DB.get_analysis(analysis_id)
        if result is None:
            raise HTTPException(404, "analysis not found")
        await asyncio.to_thread(repmod.generate, result, REPORTS_DIR)
    if not path.exists():
        raise HTTPException(500, "report generation failed")
    media = {"pdf": "application/pdf", "json": "application/json",
             "csv": "text/csv"}[fmt]
    return FileResponse(path, media_type=media, filename=path.name)


@app.post("/api/report/session/{session_id}")
async def session_report(session_id: int):
    rows = DB.list_analyses(session_id, limit=1000)
    if not rows:
        raise HTTPException(404, "no analyses in this session")
    results = [DB.get_analysis(r["analysis_id"]) for r in reversed(rows)]
    results = [r for r in results if r]
    session = DB.get_session(session_id) or {"name": f"session{session_id}"}
    path = await asyncio.to_thread(repmod.generate_batch, results, REPORTS_DIR,
                                   session["name"])
    return {"ok": True, "file": str(path), "n_analyses": len(results)}


@app.get("/api/report/session/{session_id}/download")
async def download_session_report(session_id: int):
    session = DB.get_session(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session["name"])
    path = REPORTS_DIR / f"session_{safe}_summary.pdf"
    if not path.exists():
        await session_report(session_id)
    return FileResponse(path, media_type="application/pdf", filename=path.name)


# ---------------------------------------------------------------------------
#  Live streaming
# ---------------------------------------------------------------------------
async def _stream_loop():
    dev = ST.require_device()
    while ST.streaming and ST.clients:
        try:
            counts = await asyncio.to_thread(dev.acquire, 1)
            wl = dev.wavelengths()
            stride = max(1, wl.size // 700)
            await broadcast({
                "type": "live",
                "data": {
                    "wavelength_nm": [round(float(v), 1) for v in wl[::stride]],
                    "counts": [round(float(v), 1) for v in counts[::stride]],
                    "peak_counts": float(counts.max()),
                    "saturated": bool(counts.max() >= 65000),
                    "t": time.time(),
                },
            })
        except Exception as exc:
            await broadcast({"type": "error", "message": str(exc)})
            break
        await asyncio.sleep(0.25)
    ST.streaming = False
    await broadcast({"type": "streaming", "active": False})


@app.post("/api/stream/{action}")
async def stream_control(action: str):
    if action == "start":
        if not ST.streaming:
            ST.require_device()
            ST.streaming = True
            asyncio.create_task(_stream_loop())
    elif action == "stop":
        ST.streaming = False
    else:
        raise HTTPException(400, "action must be start or stop")
    return {"ok": True, "streaming": ST.streaming}


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    # Authenticate before accepting. Starlette's http middleware does not run
    # for websocket connections, so without this check the socket is an
    # unauthenticated side door onto the same live spectra and analysis results
    # the REST API protects - which matters the moment the console is published
    # through a tunnel.
    expected = authmod.password()
    if expected:
        supplied = ws.query_params.get("token", "")
        if not (supplied and hmac.compare_digest(supplied.encode(), expected.encode())):
            await ws.close(code=1008)      # policy violation
            return

    await ws.accept()
    ST.clients.add(ws)
    try:
        if ST.last_analysis:
            await ws.send_text(json.dumps({"type": "analysis",
                                           "data": ST.last_analysis}))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ST.clients.discard(ws)


# ---------------------------------------------------------------------------
#  Frontend
# ---------------------------------------------------------------------------
FRONTEND = ROOT / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    path = FRONTEND / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    html = path.read_text(encoding="utf-8")

    # A browser cannot attach an Authorization header to a WebSocket handshake,
    # so the token travels in the query string instead. Injecting it here is
    # safe because this response is only produced for a request that already
    # passed authentication.
    pw = authmod.password()
    if pw:
        tag = '<script src="/static/app.js">'
        inject = f"<script>window.__SPECTRO_TOKEN__={json.dumps(pw)};</script>"
        html = html.replace(tag, inject + tag)
    return HTMLResponse(html)
