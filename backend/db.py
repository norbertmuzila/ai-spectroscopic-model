"""
Session and analysis persistence.

SQLite, because the whole point is that this runs on the bench next to the
spectrometer with no server to administer. Every analysis is stored in full as
JSON so a report can be regenerated months later without re-measuring, and the
spectra are kept alongside the conclusions so a future model version can be
re-run over the entire archive.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    operator     TEXT,
    site         TEXT,
    notes        TEXT,
    started_at   REAL NOT NULL,
    ended_at     REAL
);

CREATE TABLE IF NOT EXISTS analyses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id   TEXT UNIQUE NOT NULL,
    session_id    INTEGER,
    created_at    REAL NOT NULL,
    sample_label  TEXT,
    mineral       TEXT,
    confidence    REAL,
    status        TEXT,
    snr           REAL,
    source        TEXT,
    payload       TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_analyses_session ON analyses(session_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_mineral ON analyses(mineral);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.commit()
        _local.conn = conn
    return conn


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        _connect(self.path)

    @property
    def conn(self) -> sqlite3.Connection:
        return _connect(self.path)

    # ---- sessions ----------------------------------------------------
    def create_session(self, name: str, operator: str = "", site: str = "",
                       notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (name, operator, site, notes, started_at) "
            "VALUES (?,?,?,?,?)",
            (name, operator, site, notes, time.time()))
        self.conn.commit()
        return int(cur.lastrowid)

    def end_session(self, session_id: int) -> None:
        self.conn.execute("UPDATE sessions SET ended_at=? WHERE id=?",
                          (time.time(), session_id))
        self.conn.commit()

    def list_sessions(self, limit: int = 50) -> list:
        rows = self.conn.execute(
            "SELECT s.*, COUNT(a.id) AS n_analyses FROM sessions s "
            "LEFT JOIN analyses a ON a.session_id = s.id "
            "GROUP BY s.id ORDER BY s.started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: int):
        row = self.conn.execute("SELECT * FROM sessions WHERE id=?",
                                (session_id,)).fetchone()
        return dict(row) if row else None

    # ---- analyses ----------------------------------------------------
    def save_analysis(self, result: dict, session_id: int | None = None,
                      source: str = "live") -> int:
        ident = result.get("identification", {})
        cur = self.conn.execute(
            "INSERT OR REPLACE INTO analyses "
            "(analysis_id, session_id, created_at, sample_label, mineral, "
            " confidence, status, snr, source, payload) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (result["analysis_id"], session_id, result["timestamp"],
             result.get("sample_label"), ident.get("mineral"),
             ident.get("confidence"), ident.get("status"),
             result.get("quality", {}).get("snr"), source,
             json.dumps(result)))
        self.conn.commit()
        return int(cur.lastrowid)

    def get_analysis(self, analysis_id: str):
        row = self.conn.execute("SELECT payload FROM analyses WHERE analysis_id=?",
                                (analysis_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_analyses(self, session_id: int | None = None, limit: int = 100) -> list:
        if session_id is None:
            rows = self.conn.execute(
                "SELECT analysis_id, session_id, created_at, sample_label, mineral, "
                "confidence, status, snr, source FROM analyses "
                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT analysis_id, session_id, created_at, sample_label, mineral, "
                "confidence, status, snr, source FROM analyses WHERE session_id=? "
                "ORDER BY created_at DESC LIMIT ?", (session_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def delete_analysis(self, analysis_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM analyses WHERE analysis_id=?",
                                (analysis_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) n, AVG(confidence) c, AVG(snr) s FROM analyses").fetchone()
        minerals = self.conn.execute(
            "SELECT mineral, COUNT(*) n FROM analyses WHERE mineral IS NOT NULL "
            "GROUP BY mineral ORDER BY n DESC LIMIT 12").fetchall()
        statuses = self.conn.execute(
            "SELECT status, COUNT(*) n FROM analyses GROUP BY status").fetchall()
        return {
            "total_analyses": int(row["n"] or 0),
            "mean_confidence": round(float(row["c"] or 0.0), 4),
            "mean_snr": round(float(row["s"] or 0.0), 1),
            "top_minerals": [dict(r) for r in minerals],
            "status_counts": {r["status"]: r["n"] for r in statuses},
        }
