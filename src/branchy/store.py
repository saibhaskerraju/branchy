from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable


class StoreConflict(RuntimeError):
    """The stored tree version differs from what the caller expected."""


@runtime_checkable
class TreeStore(Protocol):
    def load(self, tree_id: str) -> Optional[dict]: ...
    def version(self, tree_id: str) -> int: ...
    def save_turn(
        self,
        tree_id: str,
        snapshot_meta: dict,
        new_nodes: list[dict],
        stickers: list[dict],
        expected_version: int,
    ) -> None: ...
    def list_trees(self) -> list[str]: ...
    def snapshot_all(self) -> dict: ...
    def global_version(self) -> int: ...
    def close(self) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trees(
    tree_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    version INTEGER NOT NULL,
    active_root_id TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS nodes(
    tree_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    parent_message_id TEXT,
    node_type TEXT NOT NULL,
    content TEXT NOT NULL,
    created_seq INTEGER NOT NULL,
    created_at TEXT,
    tentative INTEGER NOT NULL DEFAULT 0,
    routing TEXT,
    PRIMARY KEY(tree_id, message_id)
);
CREATE INDEX IF NOT EXISTS nodes_parent ON nodes(tree_id, parent_message_id);
CREATE TABLE IF NOT EXISTS stickers(
    tree_id TEXT NOT NULL,
    root_id TEXT NOT NULL,
    text TEXT NOT NULL,
    last_used_seq INTEGER NOT NULL,
    PRIMARY KEY(tree_id, root_id)
);
"""


class SQLiteStore:
    """SQLite-backed TreeStore; one connection per thread."""

    def __init__(self, path: str | os.PathLike = ".branchy/branchy.sqlite3") -> None:
        self.path = str(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        with conn:
            conn.executescript(_SCHEMA)

    # ---- reads ----

    def version(self, tree_id: str) -> int:
        row = self._connect().execute(
            "SELECT version FROM trees WHERE tree_id=?", (tree_id,)
        ).fetchone()
        return row[0] if row else 0

    def global_version(self) -> int:
        row = self._connect().execute(
            "SELECT COALESCE(SUM(version),0) FROM trees"
        ).fetchone()
        return int(row[0])

    def list_trees(self) -> list[str]:
        rows = self._connect().execute(
            "SELECT tree_id FROM trees ORDER BY tree_id"
        ).fetchall()
        return [r[0] for r in rows]

    def load(self, tree_id: str) -> Optional[dict]:
        conn = self._connect()
        meta = conn.execute(
            "SELECT seq, version, active_root_id FROM trees WHERE tree_id=?",
            (tree_id,),
        ).fetchone()
        if meta is None:
            return None
        seq, version, active_root_id = meta
        node_rows = conn.execute(
            "SELECT message_id, parent_message_id, node_type, content,"
            " created_seq, created_at, tentative, routing"
            " FROM nodes WHERE tree_id=? ORDER BY created_seq",
            (tree_id,),
        ).fetchall()
        children: dict[Optional[str], list[str]] = {}
        for r in node_rows:
            children.setdefault(r[1], []).append(r[0])
        nodes = [
            {
                "message_id": r[0],
                "content": r[3],
                "node_type": r[2],
                "parent_message_id": r[1],
                "child_message_ids": children.get(r[0], []),
                "created_seq": r[4],
                "created_at": r[5] or "",
                "tentative": bool(r[6]),
                "routing": json.loads(r[7]) if r[7] else None,
            }
            for r in node_rows
        ]
        sticker_rows = conn.execute(
            "SELECT root_id, text, last_used_seq FROM stickers WHERE tree_id=?",
            (tree_id,),
        ).fetchall()
        stickers = [
            {"root_id": r[0], "text": r[1], "last_used_seq": r[2]}
            for r in sticker_rows
        ]
        return {
            "schema": 1,
            "tree_id": tree_id,
            "version": version,
            "seq": seq,
            "active_root_id": active_root_id,
            "nodes": nodes,
            "stickers": stickers,
        }

    def snapshot_all(self) -> dict:
        return {
            "schema": 1,
            "trees": [s for s in (self.load(t) for t in self.list_trees()) if s],
        }

    # ---- write ----

    def save_turn(
        self,
        tree_id: str,
        snapshot_meta: dict,
        new_nodes: list[dict],
        stickers: list[dict],
        expected_version: int,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT version FROM trees WHERE tree_id=?", (tree_id,)
            ).fetchone()
            stored = row[0] if row else 0
            if stored != expected_version:
                conn.execute("ROLLBACK")
                raise StoreConflict(
                    f"tree {tree_id!r}: stored version {stored} != "
                    f"expected {expected_version}"
                )
            if row is None:
                conn.execute(
                    "INSERT INTO trees(tree_id, seq, version, active_root_id,"
                    " updated_at) VALUES(?,?,?,?,?)",
                    (tree_id, 0, 0, None, ""),
                )
            for n in new_nodes:
                conn.execute(
                    "INSERT INTO nodes(tree_id, message_id, parent_message_id,"
                    " node_type, content, created_seq, created_at, tentative,"
                    " routing) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        tree_id,
                        n["message_id"],
                        n["parent_message_id"],
                        n["node_type"],
                        n["content"],
                        n["created_seq"],
                        n.get("created_at", ""),
                        int(bool(n.get("tentative"))),
                        json.dumps(n.get("routing"))
                        if n.get("routing") is not None
                        else None,
                    ),
                )
            conn.execute("DELETE FROM stickers WHERE tree_id=?", (tree_id,))
            for s in stickers:
                conn.execute(
                    "INSERT INTO stickers(tree_id, root_id, text, last_used_seq)"
                    " VALUES(?,?,?,?)",
                    (tree_id, s["root_id"], s["text"], s["last_used_seq"]),
                )
            conn.execute(
                "UPDATE trees SET seq=?, version=?, active_root_id=?, updated_at=?"
                " WHERE tree_id=?",
                (
                    snapshot_meta.get("seq", 0),
                    expected_version + 1,
                    snapshot_meta.get("active_root_id"),
                    datetime.now(timezone.utc).isoformat(),
                    tree_id,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
