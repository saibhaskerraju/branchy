from __future__ import annotations

import re

from ..tree import ConversationTree
from .lexical import tokenize, top_keywords

_SENTENCE_END = re.compile(r"[.?!]")


def _tower_tokens(tree: ConversationTree, root_id: str) -> list[str]:
    tokens: list[str] = []
    for node in tree.thread_nodes(root_id):
        tokens.extend(tokenize(node.content))
    return tokens


def build_sticker(tree: ConversationTree, root_id: str) -> str:
    """Deterministic one-line summary: first sentence | keywords | last user line."""
    if root_id not in tree.messages:
        return ""
    nodes = tree.thread_nodes(root_id)
    root = tree.messages[root_id]

    first_sentence = root.content.strip()
    match = _SENTENCE_END.search(first_sentence)
    if match:
        first_sentence = first_sentence[: match.end()]
    first_sentence = first_sentence[:160].strip()

    other_tokens = [
        _tower_tokens(tree, other_root_id)
        for other_root_id in tree.stickers.keys() | {
            r.message_id for r in tree.get_root_messages()
        }
        if other_root_id != root_id
    ]
    keywords = top_keywords(_tower_tokens(tree, root_id), other_tokens, k=8)

    user_nodes = [n for n in nodes if n.node_type == "user"]
    last_user_line = ""
    if user_nodes:
        last_user_line = user_nodes[-1].content.strip().splitlines()[0][:160]

    return f"{first_sentence} | {', '.join(keywords)} | {last_user_line}"
