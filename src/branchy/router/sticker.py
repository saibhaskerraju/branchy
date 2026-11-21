from __future__ import annotations

import re

from ..models import MessageNode
from ..tree import ConversationTree
from .text import terms, top_keywords

_SENTENCE_END = re.compile(r"[.?!]")


def _tower_tokens(tree: ConversationTree, root_id: str) -> list[str]:
    tokens: list[str] = []
    for node in tree.thread_nodes(root_id):
        tokens.extend(terms(node.content))
    return tokens


def sticker_text(
    root: MessageNode,
    nodes: list[MessageNode],
    own_tokens: list[str],
    other_tokens: list[list[str]],
) -> str:
    """Deterministic one-line summary: first sentence | keywords | last user line.

    nodes: the topic's messages by created_seq; own_tokens: their terms;
    other_tokens: the terms of every other topic (order does not matter).
    """
    first_sentence = root.content.strip()
    match = _SENTENCE_END.search(first_sentence)
    if match:
        first_sentence = first_sentence[: match.end()]
    first_sentence = first_sentence[:160].strip()

    keywords = top_keywords(own_tokens, other_tokens, k=8)

    user_nodes = [n for n in nodes if n.node_type == "user"]
    last_user_line = ""
    if user_nodes:
        last_user_line = user_nodes[-1].content.strip().splitlines()[0][:160]

    return f"{first_sentence} | {', '.join(keywords)} | {last_user_line}"


def build_sticker(tree: ConversationTree, root_id: str) -> str:
    """The sticker for one topic, computed from scratch. Rebuilding every topic
    this way is slow (each call re-reads every other topic); refresh_stickers
    does them all at once with the same result."""
    if root_id not in tree.messages:
        return ""
    other_tokens = [
        _tower_tokens(tree, other_root_id)
        for other_root_id in tree.stickers.keys() | {
            r.message_id for r in tree.get_root_messages()
        }
        if other_root_id != root_id
    ]
    return sticker_text(
        tree.messages[root_id],
        tree.thread_nodes(root_id),
        _tower_tokens(tree, root_id),
        other_tokens,
    )
