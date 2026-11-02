from __future__ import annotations

from typing import Callable

from .core import ViewerCore

_STATUS_TEXT = {
    200: "200 OK",
    304: "304 Not Modified",
    307: "307 Temporary Redirect",
    404: "404 Not Found",
    405: "405 Method Not Allowed",
}


def wsgi_app(
    snapshot_provider: Callable[[], dict],
    refresh_seconds: int = 2,
):
    """Plain WSGI app serving the viewer."""
    from .server import _disabled

    if _disabled():

        def disabled_app(environ, start_response):
            body = b"branchy viewer disabled"
            start_response(
                "404 Not Found",
                [("Content-Type", "text/plain"),
                 ("Content-Length", str(len(body)))],
            )
            return [body]

        return disabled_app

    core = ViewerCore(snapshot_provider, refresh_seconds)

    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/") or "/"
        script_name = environ.get("SCRIPT_NAME", "") or ""
        rel = path
        if not rel.startswith("/"):
            rel = "/" + rel

        if rel in ("", "/") and not path.endswith("/"):
            target = (script_name or "") + "/"
            start_response(
                _STATUS_TEXT[307],
                [("Location", target), ("Content-Length", "0")],
            )
            return [b""]

        status, headers, body = core.handle(
            environ.get("REQUEST_METHOD", "GET"),
            rel,
            environ.get("HTTP_IF_NONE_MATCH"),
        )
        start_response(
            _STATUS_TEXT.get(status, f"{status} Status"),
            [(k.title(), v) for k, v in headers.items()],
        )
        return [body]

    return app
