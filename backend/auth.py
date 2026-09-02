"""
Access control for the console.

Off by default. On localhost the dashboard talks to a spectrometer sitting on
the same desk, and a password there is friction with no benefit.

The moment the app is published through a tunnel that changes completely. The
API can start acquisitions, take references, overwrite calibration state and
read every analysis ever stored, and a quick-tunnel hostname is a public DNS
name that gets crawled. So `run.py --tunnel` sets a password and this module
enforces it.

Enabled by setting ``SPECTRO_PASSWORD``. Requests then need either HTTP Basic
credentials or an ``X-Spectro-Token`` header; the browser gets a normal login
prompt, and scripted clients can pass the token. Comparison is constant-time so
the check cannot be turned into a timing oracle, and static assets stay open so
the login page can style itself.
"""
from __future__ import annotations

import base64
import hmac
import os
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse, Response

ENV_PASSWORD = "SPECTRO_PASSWORD"
ENV_USER = "SPECTRO_USER"
HEADER = "x-spectro-token"

# Paths that never require credentials: the health check (so a tunnel or host
# can probe liveness) and the static assets referenced by the login prompt.
OPEN_PATHS = {"/api/health", "/favicon.ico"}


def password() -> str | None:
    return os.environ.get(ENV_PASSWORD) or None


def username() -> str:
    return os.environ.get(ENV_USER, "spectro")


def generate_password(groups: int = 3, size: int = 4) -> str:
    """
    A readable, unambiguous password - it gets typed by hand off a screen.

    The alphabet drops the characters people misread aloud or in a terminal
    font: l/1/I and o/0. Three groups of four gives ~59 bits, ample for a
    hostname that only exists for the length of one session.
    """
    alphabet = "abcdefghijkmnpqrstuvwxyz23456789"
    return "-".join(
        "".join(secrets.choice(alphabet) for _ in range(size))
        for _ in range(groups)
    )


def _ok(supplied: str, expected: str) -> bool:
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


async def middleware(request: Request, call_next):
    expected = password()
    if not expected:
        return await call_next(request)

    path = request.url.path
    if path in OPEN_PATHS or path.startswith("/static/"):
        return await call_next(request)

    token = request.headers.get(HEADER)
    if token and _ok(token, expected):
        return await call_next(request)

    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8", "replace")
            user, _, pw = decoded.partition(":")
            if _ok(user, username()) and _ok(pw, expected):
                return await call_next(request)
        except Exception:
            pass

    # WebSocket upgrades cannot show a browser login prompt, so they must carry
    # the token in the query string; the page passes it along automatically.
    if path.startswith("/ws/"):
        qtoken = request.query_params.get("token", "")
        if qtoken and _ok(qtoken, expected):
            return await call_next(request)
        return JSONResponse({"detail": "unauthorized"}, status_code=401)

    if path.startswith("/api/"):
        return JSONResponse(
            {"detail": "Authentication required. Use the password shown in the "
                       "terminal that started the tunnel."},
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Spectral Console"'},
        )

    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Spectral Console"'},
        content="Authentication required.",
        media_type="text/plain",
    )
