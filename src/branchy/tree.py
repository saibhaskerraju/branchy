from __future__ import annotations

from typing import Any, Optional
import uuid

from .models import MessageNode


class ConversationTree:
    def __init__(self) -> None:
        self.messages: dict[str, MessageNode] = {}

    def add_message(self, user: str, assistant: str, parent_message_id: Optional[str] = None) -> str:
        message_id = str(uuid.uuid4())[:6]
        message = MessageNode(
            message_id=message_id,
            user=user,
            assistant=assistant,
            parent_message_id=parent_message_id,
        )
        self.messages[message_id] = message

        if parent_message_id and parent_message_id in self.messages:
            self.messages[parent_message_id].child_message_ids.append(
                message_id)

        return message_id

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

        for message_id in self.messages:
            full_path = self.get_path_to_root(message_id)

            for window_size in range(1, min(len(full_path), max_window) + 1):
                window = full_path[-window_size:]

                trace_lines = []
                for message in window:
                    trace_lines.append(f"User: {message.user}")
                    trace_lines.append(f"Assistant: {message.assistant}")
                    trace_lines.append("")

                traces.append(
                    {
                        "trace_id": f"{message_id}_w{window_size}",
                        "end_message_id": message_id,
                        "window_size": window_size,
                        "messages": window,
                        "formatted_text": "\n".join(trace_lines),
                    }
                )

        return traces

    def get_root_messages(self) -> list[MessageNode]:
        return [message for message in self.messages.values() if message.parent_message_id is None]

    def render_tree(self) -> str:
        if not self.messages:
            return "Empty tree"

        lines = []
        roots = self.get_root_messages()

        for root_message in roots:
            lines.extend(self._render_message(root_message, "", is_last=True))

        return "\n".join(lines)

    def _render_message(self, message: MessageNode, prefix: str, is_last: bool) -> list[str]:
        branch = "└── " if is_last else "├── "
        header = f"{prefix}{branch}[{message.message_id}] User: {message.user}"
        lines = [header]

        next_prefix = prefix + ("    " if is_last else "│   ")
        for index, child_id in enumerate(message.child_message_ids):
            child = self.messages[child_id]
            child_is_last = index == len(message.child_message_ids) - 1
            lines.extend(self._render_message(
                child, next_prefix, child_is_last))

        return lines
