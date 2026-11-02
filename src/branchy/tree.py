from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
import threading
import uuid

from .models import MessageNode, Sticker


class ConversationTree:
    def __init__(self, tree_id: str = "default") -> None:
        self.tree_id = tree_id
        self.messages: dict[str, MessageNode] = {}
        self.seq: int = 0
        self.stickers: dict[str, Sticker] = {}
        self.active_root_id: Optional[str] = None
        self.version: int = 0
        self._lock = threading.RLock()

    def add_message(
        self,
        user: str,
        assistant: str,
        parent_message_id: Optional[str] = None,
        tentative: bool = False,
        routing: Optional[dict] = None,
    ) -> str:
        with self._lock:
            # Create user node
            self.seq += 1
            user_id = str(uuid.uuid4())
            user_node = MessageNode(
                message_id=user_id,
                content=user,
                node_type="user",
                parent_message_id=parent_message_id,
                created_seq=self.seq,
                created_at=datetime.now(timezone.utc).isoformat(),
                tentative=tentative,
                routing=routing,
            )
            self.messages[user_id] = user_node

            if parent_message_id and parent_message_id in self.messages:
                self.messages[parent_message_id].child_message_ids.append(user_id)

            # Create assistant node as child of user node
            self.seq += 1
            assistant_id = str(uuid.uuid4())
            assistant_node = MessageNode(
                message_id=assistant_id,
                content=assistant,
                node_type="assistant",
                parent_message_id=user_id,
                created_seq=self.seq,
                created_at=datetime.now(timezone.utc).isoformat(),
                tentative=tentative,
            )
            self.messages[assistant_id] = assistant_node
            user_node.child_message_ids.append(assistant_id)

            self.active_root_id = self.root_of(user_id)
            self.version += 1
            return user_id

    def root_of(self, message_id: str) -> str:
        """Walk up to the root user node containing message_id."""
        current_id: Optional[str] = message_id
        while current_id and current_id in self.messages:
            node = self.messages[current_id]
            if node.parent_message_id is None:
                return node.message_id
            current_id = node.parent_message_id
        return message_id

    def leaf_assistants(self) -> list[MessageNode]:
        """Assistant nodes with no children, ordered by created_seq."""
        return sorted(
            (
                m
                for m in self.messages.values()
                if m.node_type == "assistant" and not m.child_message_ids
            ),
            key=lambda m: m.created_seq,
        )

    def root_assistants(self) -> list[MessageNode]:
        """The assistant child of each root user node, ordered by created_seq."""
        assistants = []
        for root in self.get_root_messages():
            for child_id in root.child_message_ids:
                child = self.messages[child_id]
                if child.node_type == "assistant":
                    assistants.append(child)
        return sorted(assistants, key=lambda m: m.created_seq)

    def thread_nodes(self, root_id: str) -> list[MessageNode]:
        """All nodes in the tower rooted at root_id, ordered by created_seq."""
        nodes = [
            m for m in self.messages.values() if self.root_of(m.message_id) == root_id
        ]
        return sorted(nodes, key=lambda m: m.created_seq)

    def last_turn(self, assistant_node: MessageNode) -> tuple[str, str]:
        """(user content, assistant content) for the pair ending at assistant_node."""
        user_content = ""
        parent_id = assistant_node.parent_message_id
        if parent_id and parent_id in self.messages:
            user_content = self.messages[parent_id].content
        return user_content, assistant_node.content

    def refresh_stickers(self) -> None:
        """Refresh every tower's sticker text."""
        from .router.sticker import build_sticker

        with self._lock:
            for root in self.get_root_messages():
                text = build_sticker(self, root.message_id)
                existing = self.stickers.get(root.message_id)
                last_used = existing.last_used_seq if existing else 0
                if root.message_id == self.active_root_id:
                    last_used = self.seq
                self.stickers[root.message_id] = Sticker(
                    root_id=root.message_id,
                    text=text,
                    last_used_seq=last_used,
                )

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot of the whole tree."""
        nodes = [
            n.model_dump()
            for n in sorted(self.messages.values(), key=lambda m: m.created_seq)
        ]
        stickers = [s.model_dump() for s in self.stickers.values()]
        return {
            "schema": 1,
            "tree_id": self.tree_id,
            "version": self.version,
            "seq": self.seq,
            "active_root_id": self.active_root_id,
            "nodes": nodes,
            "stickers": stickers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTree":
        if data.get("schema") != 1:
            raise ValueError(f"unsupported snapshot schema: {data.get('schema')!r}")
        tree = cls(tree_id=data.get("tree_id", "default"))
        tree.messages = {
            node["message_id"]: MessageNode(**node) for node in data["nodes"]
        }
        tree.stickers = {
            sticker["root_id"]: Sticker(**sticker)
            for sticker in data.get("stickers", [])
        }
        tree.seq = data["seq"]
        tree.version = data["version"]
        tree.active_root_id = data["active_root_id"]
        return tree

    def snapshot(self) -> dict:
        """to_dict() under the lock — what the viewer serves."""
        with self._lock:
            return self.to_dict()

    def get_path_to_root(self, message_id: str) -> list[MessageNode]:
        path: list[MessageNode] = []
        current_id = message_id

        while current_id and current_id in self.messages:
            message = self.messages[current_id]
            path.insert(0, message)
            current_id = message.parent_message_id

        return path

    def get_root_messages(self) -> list[MessageNode]:
        return [m for m in self.messages.values() if m.node_type == "user" and m.parent_message_id is None]

    def render_tree(self) -> str:
        if not self.messages:
            return "Empty tree"

        lines = []
        roots = self.get_root_messages()

        for root_message in roots:
            lines.extend(self._render_message(root_message, "", is_last=True))

        return "\n".join(lines)

    def render_subtree(self, message_id: str) -> str:
        """Render a subtree starting from a given message_id and all its child branches."""
        if message_id not in self.messages:
            return f"Message ID '{message_id}' not found"

        message = self.messages[message_id]
        lines = self._render_message(message, "", is_last=True)
        return "\n".join(lines)

    def _render_message(self, message: MessageNode, prefix: str, is_last: bool) -> list[str]:
        branch = "└── " if is_last else "├── "
        label = "User" if message.node_type == "user" else "Assistant"
        header = f"{prefix}{branch}[{message.message_id}] {label}: {message.content}"
        lines = [header]

        next_prefix = prefix + ("    " if is_last else "│   ")
        for index, child_id in enumerate(message.child_message_ids):
            child = self.messages[child_id]
            child_is_last = index == len(message.child_message_ids) - 1
            lines.extend(self._render_message(
                child, next_prefix, child_is_last))

        return lines
