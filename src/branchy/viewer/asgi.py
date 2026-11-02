from __future__ import annotations

from typing import Callable

from .core import ViewerCore


def asgi_app(
    snapshot_provider: Callable[[], dict],
    refresh_seconds: int = 2,
):
    """Pure ASGI 3 app serving the viewer (no framework imports)."""
    from .server import _disabled

    if _disabled():

        async def disabled_app(scope, receive, send):
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            if scope["type"] != "http":
                return
            body = b"branchy viewer disabled"
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [(b"content-type", b"text/plain"),
                                (b"content-length", str(len(body)).encode())],
                }
            )
            await send({"type": "http.response.body", "body": body})

        return disabled_app

    core = ViewerCore(snapshot_provider, refresh_seconds)

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            # reply so hosts that run lifespan on mounted apps don't hang
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return

        path = scope.get("path", "/")
        root_path = scope.get("root_path", "") or ""
        rel = path[len(root_path):] if root_path and path.startswith(root_path) else path
        if not rel.startswith("/"):
            rel = "/" + rel

        # mount root without trailing slash -> redirect so relative URLs work
        if rel in ("", "/") and not path.endswith("/"):
            target = (root_path or "") + "/"
            await send(
                {
                    "type": "http.response.start",
                    "status": 307,
                    "headers": [(b"location", target.encode()),
                                (b"content-length", b"0")],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        headers_in = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        status, headers, body = core.handle(
            scope["method"], rel, headers_in.get("if-none-match")
        )
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (k.encode("latin-1"), v.encode("latin-1"))
                    for k, v in headers.items()
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return app
