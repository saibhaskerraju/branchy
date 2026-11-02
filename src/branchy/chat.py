from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional, Union

from .llm import LiteLLMResponder, Responder
from .store import SQLiteStore, StoreConflict, TreeStore

logger = logging.getLogger("branchy")

DEFAULT_VIEWER_HOST = "127.0.0.1"
DEFAULT_VIEWER_PORT = 8787
DEFAULT_VIEWER_REFRESH_SECONDS = 2
from .router import RouterConfig, RoutingDecision, ThreadRouter
from .tree import ConversationTree


class BranchingChat:
    """Branch-aware chat manager that routes messages into a conversation tree."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        responder: Optional[Responder] = None,
        router_config: Optional[RouterConfig] = None,
        system_prompt: Optional[str] = None,
        store: Union[TreeStore, str, os.PathLike, None, Literal["default"]] = "default",
        tree_id: str = "default",
        viewer: bool = False,
        viewer_policy: Literal["auto", "allow", "deny"] = "auto",
        viewer_host: str = DEFAULT_VIEWER_HOST,
        viewer_port: int = DEFAULT_VIEWER_PORT,
        viewer_refresh_seconds: int = DEFAULT_VIEWER_REFRESH_SECONDS,
    ) -> None:
        """Create a chat instance.

        responder: any (history, user_message) -> reply callable where history
        is a list of {"role", "content"} turns. When not given, `model` is
        required and a LiteLLMResponder is built (provider prefixes and
        api_base follow litellm's rules).
        """

        if responder is not None:
            self.responder = responder
        else:
            if model is None:
                raise ValueError(
                    "model is required when no responder callable is given"
                )
            kwargs = {}
            if api_base is not None:
                kwargs["api_base"] = api_base
            if system_prompt is not None:
                kwargs["system_prompt"] = system_prompt
            self.responder = LiteLLMResponder(
                model=model, api_key=api_key, **kwargs
            )

        if isinstance(store, TreeStore):
            self.store: Optional[TreeStore] = store
        elif store is None:
            self.store = None
        elif store == "default":
            self.store = SQLiteStore(".branchy/branchy.sqlite3")
        else:
            self.store = SQLiteStore(store)

        if self.store is not None:
            snap = self.store.load(tree_id)
            self.tree = (
                ConversationTree.from_dict(snap)
                if snap
                else ConversationTree(tree_id=tree_id)
            )
            logger.info(
                "branchy store: %s (tree '%s', %d nodes loaded)",
                getattr(self.store, "path", repr(self.store)),
                tree_id,
                len(self.tree.messages),
            )
        else:
            self.tree = ConversationTree(tree_id=tree_id)

        self.router = ThreadRouter(config=router_config or RouterConfig())
        self.last_decision: Optional[RoutingDecision] = None

        self._viewer = None
        self._viewer_host = viewer_host
        self._viewer_port = viewer_port
        self._viewer_refresh = viewer_refresh_seconds
        self._viewer_policy = viewer_policy
        if viewer:
            self.start_viewer()

    def _snapshot_provider(self):
        if self.store is not None:
            return self.store.snapshot_all
        return self.tree.snapshot

    def start_viewer(self) -> None:
        """Start the standalone HTML tree viewer (idempotent).

        Policy: 'deny' never starts; 'auto' suppresses the standalone server
        when a web server process is detected; 'allow' always starts.
        """
        if self._viewer is not None:
            return
        if self._viewer_policy == "deny":
            logger.info(
                "branchy viewer: standalone viewer denied by "
                "viewer_policy='deny'"
            )
            return
        if self._viewer_policy == "auto":
            from .viewer.server import _web_server_detected

            if _web_server_detected():
                logger.warning(
                    "branchy viewer: standalone viewer suppressed — web "
                    "server process detected; mount chat.viewer_app() "
                    "instead (or viewer_policy='allow')"
                )
                return
        from .viewer import TreeViewer

        self._viewer = TreeViewer(
            self._snapshot_provider(),
            host=self._viewer_host,
            port=self._viewer_port,
            refresh_seconds=self._viewer_refresh,
        )
        self._viewer.start()
        print(f"branchy viewer: {self._viewer.url}", file=sys.stderr)

    def stop_viewer(self) -> None:
        """Stop the viewer server if running."""
        if self._viewer is not None:
            self._viewer.stop()
            self._viewer = None

    def viewer_app(self, kind: Literal["asgi", "wsgi"] = "asgi"):
        """A mountable viewer app bound to this tree's snapshot.

        Mount it inside your web app instead of running the standalone
        server, e.g. app.mount('/branchy', chat.viewer_app()).
        """
        if kind == "asgi":
            from .viewer import asgi_app

            app = asgi_app(self._snapshot_provider(), self._viewer_refresh)
        elif kind == "wsgi":
            from .viewer import wsgi_app

            app = wsgi_app(self._snapshot_provider(), self._viewer_refresh)
        else:
            raise ValueError(f"unknown viewer app kind: {kind!r}")
        logger.info(
            "branchy viewer app created; mount it on your web app "
            "(e.g. app.mount('/branchy', chat.viewer_app()))"
        )
        return app

    def close(self) -> None:
        """Release resources (the viewer server and store connections)."""
        self.stop_viewer()
        if self.store is not None:
            self.store.close()

    @property
    def viewer_url(self) -> Optional[str]:
        return self._viewer.url if self._viewer is not None else None

    def export_viewer_html(self, path) -> None:
        """Write a self-contained static HTML snapshot of the tree."""
        from .viewer import render_static_html

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            render_static_html(self.tree.snapshot()), encoding="utf-8"
        )

    def add_message(
        self,
        user_message: str,
        assistant_message: Optional[str] = None,
    ) -> str:
        """Add a user message, route it in the tree, and return the new message id."""

        self._sync()
        decision = self.router.route(self.tree, user_message)

        if assistant_message is None:
            history: list[dict] = []
            if decision.target_message_id:
                for node in self.tree.get_path_to_root(decision.target_message_id):
                    history.append(
                        {"role": node.node_type, "content": node.content}
                    )
            assistant_message = self.responder(history, user_message)

        message_id = self._add_and_persist(
            user_message, assistant_message, decision
        )
        self.last_decision = decision
        return message_id

    def _sync(self) -> None:
        """Reload the in-memory tree when the store moved ahead (other
        processes/workers sharing the file)."""
        if self.store is None:
            return
        if self.store.version(self.tree.tree_id) != self.tree.version:
            snap = self.store.load(self.tree.tree_id)
            if snap:
                self.tree = ConversationTree.from_dict(snap)

    def _add_and_persist(
        self,
        user_message: str,
        assistant_message: str,
        decision: RoutingDecision,
    ) -> str:
        expected = self.tree.version
        message_id = self.tree.add_message(
            user=user_message,
            assistant=assistant_message,
            parent_message_id=decision.target_message_id,
            tentative=decision.action == "ATTACH_TENTATIVE",
            routing=decision.model_dump(),
        )
        self.tree.refresh_stickers()
        if self.store is None:
            return message_id
        user_node = self.tree.messages[message_id]
        assistant_node = self.tree.messages[user_node.child_message_ids[0]]
        try:
            self.store.save_turn(
                self.tree.tree_id,
                {
                    "seq": self.tree.seq,
                    "active_root_id": self.tree.active_root_id,
                },
                [user_node.model_dump(), assistant_node.model_dump()],
                [s.model_dump() for s in self.tree.stickers.values()],
                expected_version=expected,
            )
        except StoreConflict:
            # someone else wrote first: reload, re-route, reuse the reply,
            # retry once
            snap = self.store.load(self.tree.tree_id)
            if snap:
                self.tree = ConversationTree.from_dict(snap)
            decision = self.router.route(self.tree, user_message)
            self.last_decision = decision
            expected = self.tree.version
            message_id = self.tree.add_message(
                user=user_message,
                assistant=assistant_message,
                parent_message_id=decision.target_message_id,
                tentative=decision.action == "ATTACH_TENTATIVE",
                routing=decision.model_dump(),
            )
            self.tree.refresh_stickers()
            user_node = self.tree.messages[message_id]
            assistant_node = self.tree.messages[user_node.child_message_ids[0]]
            self.store.save_turn(
                self.tree.tree_id,
                {
                    "seq": self.tree.seq,
                    "active_root_id": self.tree.active_root_id,
                },
                [user_node.model_dump(), assistant_node.model_dump()],
                [s.model_dump() for s in self.tree.stickers.values()],
                expected_version=expected,
            )
        return message_id

    def show_tree(self) -> str:
        """Render the current user-message tree."""

        tree_output = self.tree.render_tree()
        return tree_output

    def show_subtree(self, message_id: str) -> str:
        """Render a subtree starting from a given message_id and all its child branches."""

        tree_output = self.tree.render_subtree(message_id)
        return tree_output
