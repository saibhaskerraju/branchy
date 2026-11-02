from __future__ import annotations

import json
from typing import Callable, Optional

from .page import HTML

_REFRESH_PLACEHOLDER = "__REFRESH_SECONDS__"
_NEW_CONVERSATION = "(none - this message started a new thread)"


def _normalise(snapshot: dict) -> dict:
    """Accept either a single schema-1 tree snapshot or {"trees": [...]} and
    return the multi-tree form."""
    if "trees" in snapshot:
        return snapshot
    tree = dict(snapshot)
    tree.setdefault("tree_id", "default")
    return {"schema": 1, "trees": [tree]}


def _subtree(nodes: dict, node_id: str) -> dict:
    """Nested dict of a node and all descendants."""
    node = nodes[node_id]
    return {
        "id": node["message_id"],
        "type": node["node_type"],
        "content": node["content"],
        "tentative": node.get("tentative", False),
        "children": [
            _subtree(nodes, cid) for cid in node.get("child_message_ids", [])
            if cid in nodes
        ],
    }


def _node_detail(multi: dict, node_id: str) -> Optional[dict]:
    """Build the /api/node payload from a (normalised) snapshot dict —
    searches across all trees; storage-agnostic."""
    node = None
    nodes = {}
    stickers_list = []
    tree_id = None
    for tree in multi.get("trees", []):
        cand = {n["message_id"]: n for n in tree.get("nodes", [])}
        if node_id in cand:
            nodes = cand
            node = cand[node_id]
            stickers_list = tree.get("stickers", [])
            tree_id = tree.get("tree_id")
            break
    if node is None:
        return None

    path = []
    current = node
    while current is not None:
        path.insert(
            0,
            {
                "id": current["message_id"],
                "type": current["node_type"],
                "content": current["content"],
            },
        )
        current = nodes.get(current["parent_message_id"])

    stickers = {s["root_id"]: s for s in stickers_list}
    root = path[0]["id"] if path else None
    sticker = stickers.get(root)

    history = _NEW_CONVERSATION
    if node["node_type"] == "user" and node["parent_message_id"] in nodes:
        parent_path = []
        current = nodes[node["parent_message_id"]]
        while current is not None:
            parent_path.insert(0, current)
            current = nodes.get(current["parent_message_id"])
        history = "\n".join(
            f"{'User' if n['node_type'] == 'user' else 'Assistant'}: {n['content']}"
            for n in parent_path
        )

    return {
        "node": node,
        "tree_id": tree_id,
        "path_to_root": path,
        "sticker": sticker,
        "history": history,
        "subtree": _subtree(nodes, node_id),
    }


def render_static_html(snapshot: dict) -> str:
    """The viewer page with the snapshot inlined; no fetches at runtime."""
    return HTML.replace(
        "</body>",
        '<script id="snapshot" type="application/json">'
        + json.dumps(snapshot).replace("</", "<\\/")
        + "</script>\n</body>",
    ).replace(_REFRESH_PLACEHOLDER, "0")


class ViewerCore:
    """Framework-neutral request handling shared by every viewer front-end.

    Paths are relative to the mount point: front-ends strip any prefix before
    calling handle().
    """

    def __init__(
        self,
        snapshot_provider: Callable[[], dict],
        refresh_seconds: int = 2,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.page = HTML.replace(_REFRESH_PLACEHOLDER, str(refresh_seconds)).encode()

    def handle(
        self,
        method: str,
        path: str,
        if_none_match: Optional[str] = None,
    ) -> tuple[int, dict, bytes]:
        """(status, headers, body) for a mount-relative path."""
        method = method.upper()
        if method not in ("GET", "HEAD"):
            return 405, {"content-type": "text/plain"}, b"method not allowed"
        body_only_head = method == "HEAD"

        if path in ("", "/"):
            return self._respond(
                200,
                {"content-type": "text/html; charset=utf-8",
                 "cache-control": "no-store"},
                self.page,
                body_only_head,
            )
        if path in ("/api/tree", "/api/trees"):
            snapshot = _normalise(self.snapshot_provider())
            global_version = sum(
                t.get("version", 0) for t in snapshot["trees"]
            )
            etag = f'"{global_version}"'
            headers = {
                "content-type": "application/json",
                "cache-control": "no-store",
                "etag": etag,
            }
            if if_none_match == etag:
                return 304, headers, b""
            return self._respond(
                200, headers, json.dumps(snapshot).encode(), body_only_head
            )
        if path.startswith("/api/node/"):
            node_id = path[len("/api/node/"):]
            detail = _node_detail(_normalise(self.snapshot_provider()), node_id)
            if detail is None:
                return self._respond(
                    404,
                    {"content-type": "application/json"},
                    json.dumps({"error": "node not found"}).encode(),
                    body_only_head,
                )
            return self._respond(
                200,
                {"content-type": "application/json",
                 "cache-control": "no-store"},
                json.dumps(detail).encode(),
                body_only_head,
            )
        return self._respond(
            404, {"content-type": "text/plain"}, b"not found", body_only_head
        )

    @staticmethod
    def _respond(
        status: int, headers: dict, body: bytes, head_only: bool
    ) -> tuple[int, dict, bytes]:
        headers = dict(headers)
        headers["content-length"] = str(len(body))
        return status, headers, b"" if head_only else body
