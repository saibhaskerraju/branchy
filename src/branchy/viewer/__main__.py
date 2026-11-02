"""External viewer: python -m branchy.viewer --db PATH [--host H] [--port P] [--refresh S]

Serves the standalone viewer over a SQLiteStore snapshot — useful for viewing
a DB on a mounted volume while the app runs elsewhere (e.g. in Docker).
"""

from __future__ import annotations

import argparse

from ..store import SQLiteStore
from .server import TreeViewer


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m branchy.viewer")
    parser.add_argument("--db", required=True, help="path to branchy sqlite db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--refresh", type=int, default=2)
    args = parser.parse_args()

    store = SQLiteStore(args.db)
    viewer = TreeViewer(
        store.snapshot_all,
        host=args.host,
        port=args.port,
        refresh_seconds=args.refresh,
    )
    viewer.start()
    print(f"branchy viewer: {viewer.url} (db: {args.db}) — Ctrl+C to stop")
    try:
        viewer._thread.join()  # block until interrupted
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()
        store.close()


if __name__ == "__main__":
    main()
