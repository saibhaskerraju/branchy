from __future__ import annotations

from typing import Any, Optional
import uuid

from .models import MessageNode


class ConversationTree:
    def __init__(self) -> None:
        self.messages: dict[str, MessageNode] = {}

    def add_message(self, user: str, assistant: str, parent_message_id: Optional[str] = None) -> str:
        # Create user node
        user_id = str(uuid.uuid4())
        user_node = MessageNode(
            message_id=user_id,
            content=user,
            node_type="user",
            parent_message_id=parent_message_id,
        )
        self.messages[user_id] = user_node

        if parent_message_id and parent_message_id in self.messages:
            self.messages[parent_message_id].child_message_ids.append(user_id)

        # Create assistant node as child of user node
        assistant_id = str(uuid.uuid4())
        assistant_node = MessageNode(
            message_id=assistant_id,
            content=assistant,
            node_type="assistant",
            parent_message_id=user_id,
        )
        self.messages[assistant_id] = assistant_node
        user_node.child_message_ids.append(assistant_id)

        return user_id

    def get_path_to_root(self, message_id: str) -> list[MessageNode]:
        path: list[MessageNode] = []
        current_id = message_id

        while current_id and current_id in self.messages:
            message = self.messages[current_id]
            path.insert(0, message)
            current_id = message.parent_message_id

        return path

    def get_all_traces(self, max_window: int = 3) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []

        user_nodes = [m for m in self.messages.values() if m.node_type == "user"]

        for user_node in user_nodes:
            # Path from root to this user node (alternating user/assistant)
            full_path = self.get_path_to_root(user_node.message_id)

            # Append the assistant child of this user node to complete the last pair
            if user_node.child_message_ids:
                full_path.append(self.messages[user_node.child_message_ids[0]])

            # Group into (user, assistant) pairs
            pairs: list[tuple[Any, Any]] = []
            i = 0
            while i < len(full_path) - 1:
                if full_path[i].node_type == "user" and full_path[i + 1].node_type == "assistant":
                    pairs.append((full_path[i], full_path[i + 1]))
                    i += 2
                else:
                    i += 1

            if not pairs:
                continue

            for window_size in range(1, min(len(pairs), max_window) + 1):
                window_pairs = pairs[-window_size:]

                trace_lines = []
                for user_msg, asst_msg in window_pairs:
                    trace_lines.append(f"User: {user_msg.content}")
                    trace_lines.append(f"Assistant: {asst_msg.content}")
                    trace_lines.append("")

                traces.append(
                    {
                        "trace_id": f"{user_node.message_id}_w{window_size}",
                        "end_message_id": window_pairs[-1][1].message_id,
                        "root_assistant_id": window_pairs[0][1].message_id,
                        "window_size": window_size,
                        "messages": [p[0] for p in window_pairs],
                        "formatted_text": "\n".join(trace_lines),
                    }
                )

        return traces

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
