from __future__ import annotations

import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from urllib.parse import urlparse

from .core import ViewerCore

logger = logging.getLogger("branchy.viewer")

_LOOPBACK = ("127.0.0.1", "localhost", "::1")

_WEB_SERVERS = frozenset(
    {"uvicorn", "gunicorn", "hypercorn", "waitress", "daphne", "werkzeug.serving"}
)


def _disabled() -> bool:
    """BRANCHY_VIEWER_DISABLE — the only env var the library reads; it can
    only turn things off."""
    import os

    return os.environ.get("BRANCHY_VIEWER_DISABLE", "").lower() in ("1", "true")


def _web_server_detected() -> bool:
    """True inside a web-server process: a known server module is imported,
    or an asyncio loop is running (mounted apps shouldn't get a second port)."""
    import asyncio
    import sys

    if any(name in sys.modules for name in _WEB_SERVERS):
        return True
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _in_container() -> bool:
    """Best-effort container detection; never raises."""
    try:
        if os.path.exists("/.dockerenv"):
            return True
        with open("/proc/1/cgroup", "rb") as fh:
            data = fh.read()
        return any(tag in data for tag in (b"docker", b"kubepods", b"containerd"))
    except Exception:
        return False


class _ExclusiveHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False


class TreeViewer:
    """Serve a live HTML view of a conversation-tree snapshot.

    Standalone mode for scripts/notebooks/console apps: a daemon-thread HTTP
    server. Never raises for bind problems — falls back to a free port or
    stays stopped.
    """

    def __init__(
        self,
        snapshot_provider: Callable[[], dict],
        host: str = "127.0.0.1",
        port: int = 8787,
        refresh_seconds: int = 2,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.host = host
        self.port = port
        self.refresh_seconds = refresh_seconds
        self._core = ViewerCore(snapshot_provider, refresh_seconds)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> Optional[str]:
        if self._server is None:
            return None
        return f"http://{self.host}:{self._server.server_address[1]}"

    def start(self) -> None:
        if self._server is not None:
            return
        if _disabled():
            logger.warning(
                "branchy viewer: disabled by BRANCHY_VIEWER_DISABLE"
            )
            return
        core = self._core

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence per-request logging
                pass

            def _dispatch(self):
                path = urlparse(self.path).path
                status, headers, body = core.handle(
                    self.command, path, self.headers.get("if-none-match")
                )
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                if body:
                    self.wfile.write(body)

            do_GET = _dispatch
            do_HEAD = _dispatch
            do_POST = _dispatch
            do_PUT = _dispatch
            do_DELETE = _dispatch

        actual_port = self.port
        try:
            self._server = _ExclusiveHTTPServer((self.host, actual_port), Handler)
        except OSError:
            try:
                self._server = _ExclusiveHTTPServer((self.host, 0), Handler)
                logger.warning(
                    "branchy viewer: port %s busy, serving on %s",
                    self.port,
                    f"http://{self.host}:{self._server.server_address[1]}",
                )
            except OSError as exc:
                logger.warning(
                    "branchy viewer: could not bind any port (%s); not started",
                    exc,
                )
                self._server = None
                return

        if self.host in _LOOPBACK and _in_container():
            bound_port = self._server.server_address[1]
            logger.warning(
                "branchy viewer is bound to %s:%s inside a container and is "
                "NOT reachable from your host browser. Either mount it into "
                "your web app (app.mount('/branchy', chat.viewer_app())) or "
                "pass viewer_host='0.0.0.0' and publish the port in compose "
                "(ports: ['%s:%s']). The viewer has no authentication.",
                self.host,
                bound_port,
                bound_port,
                bound_port,
            )

        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="branchy-viewer"
        )
        self._thread.start()
        logger.info("branchy viewer: %s", self.url)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
