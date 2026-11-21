"""Human-readable views of a conversation tree and its routing decisions.

Messages get short labels: U<n> is the n-th user message (in creation order),
A<n> is its reply. Used by BranchingChat.show_tree(), describe() and explain().
"""

from __future__ import annotations

from typing import Optional

from .router import RoutingDecision
from .tree import ConversationTree

ACTION_ICON = {"NEW": "★", "CONTINUE": "↳", "SWITCH": "↪", "FORK": "⑂"}
_TREE_MARK = {"NEW": " ★", "FORK": " ⑂", "SWITCH": " ↪"}


def labels(tree: ConversationTree) -> dict[str, str]:
    """message_id -> "U<n>" / "A<n>"."""
    users = sorted(
        (m for m in tree.messages.values() if m.node_type == "user"),
        key=lambda m: m.created_seq,
    )
    out: dict[str, str] = {}
    for n, u in enumerate(users, start=1):
        out[u.message_id] = f"U{n}"
        for c in u.child_message_ids:
            out[c] = f"A{n}"
    return out


def message_id(tree: ConversationTree, label: str) -> Optional[str]:
    """The message id for a label like "A3" (case-insensitive), or None."""
    wanted = label.strip().upper()
    return next((mid for mid, lab in labels(tree).items() if lab == wanted), None)


def active_leaf(tree: ConversationTree) -> Optional[str]:
    """The reply to the newest user message."""
    users = [m for m in tree.messages.values() if m.node_type == "user"]
    if not users:
        return None
    newest = max(users, key=lambda m: m.created_seq)
    return newest.child_message_ids[0] if newest.child_message_ids else None


def _snippet(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def render_tree(
    tree: ConversationTree, root_id: Optional[str] = None, width: int = 70
) -> str:
    """The tree under one synthetic "(conversation)" root (or the subtree of
    root_id), with labels, placement marks, tentative flags and the active leaf."""
    if not tree.messages:
        return "(empty)"
    names = labels(tree)
    active = active_leaf(tree)
    lines: list[str] = []

    def walk(mid: str, prefix: str, last: bool, top: bool = False) -> None:
        n = tree.messages[mid]
        marks = ""
        if n.node_type == "user" and n.routing:
            marks += _TREE_MARK.get(n.routing.get("action", ""), "")
            if n.tentative:
                marks += " (tentative)"
        if mid == active:
            marks += "  ← active"
        branch = "" if top else ("└── " if last else "├── ")
        lines.append(f"{prefix}{branch}{names.get(mid, '?')} {_snippet(n.content, width)}{marks}")
        kids = n.child_message_ids
        child_prefix = prefix if top else prefix + ("    " if last else "│   ")
        for i, c in enumerate(kids):
            walk(c, child_prefix, i == len(kids) - 1)

    if root_id is not None:
        if root_id not in tree.messages:
            return f"message {root_id!r} not found"
        walk(root_id, "", True, top=True)
        return "\n".join(lines)
    lines.append("(conversation)")
    roots = sorted(tree.get_root_messages(), key=lambda m: m.created_seq)
    for i, r in enumerate(roots):
        walk(r.message_id, "", i == len(roots) - 1)
    return "\n".join(lines)


def describe(
    tree: ConversationTree,
    decision: RoutingDecision,
    placed_message_id: Optional[str] = None,
) -> str:
    """One line: where the message went, e.g. "⑂ fork under A1 (sister of U2)".
    placed_message_id: the message the decision placed, if it is already in the
    tree (so it is not listed as its own sister)."""
    names = labels(tree)
    target = names.get(decision.target_message_id or "", "?")
    if decision.action == "NEW":
        text = "new topic"
    elif decision.action == "CONTINUE":
        text = f"continue under {target}"
    elif decision.action == "SWITCH":
        text = f"switch to {target}"
    else:
        sisters = [
            names.get(c, "?")
            for c in tree.messages[decision.target_message_id].child_message_ids
            if c != placed_message_id
        ]
        text = f"fork under {target}" + (f" (sister of {', '.join(sisters)})" if sisters else "")
    if decision.decided_by == "explicit":
        text += " [explicit]"
    elif decision.llm is not None:
        r = decision.llm
        text += (" [llm failed, rules kept]" if r.error else
                 " [llm agreed]" if r.agreed else
                 f" [llm overrode rules: {r.router_action.lower()} under "
                 f"{labels(tree).get(r.router_target or '', 'new topic')}]")
    if decision.tentative:
        text += " (tentative)"
    return f"{ACTION_ICON[decision.action]} {text}"


def explain(tree: ConversationTree, decision: Optional[RoutingDecision]) -> str:
    """The decision in full: reason, cues, every check, thread and node scores."""
    if decision is None:
        return "no routing yet"
    names = labels(tree)

    def name(mid: Optional[str]) -> str:
        return names.get(mid or "", "—")

    out = [
        f"{decision.action} → {name(decision.target_message_id)}  "
        f"{decision.confidence}  decided_by={decision.decided_by}",
        f"reason: {decision.reason}",
    ]
    if decision.uncertainty:
        out.append(f"uncertainty: {decision.uncertainty}")
    if decision.llm is not None:
        r = decision.llm
        opts = ", ".join(name(t) if t else "NEW" for t in r.options)
        out.append(
            f"llm review ({r.model or 'custom'}, {r.latency_ms} ms): answer {r.answer!r} "
            f"-> {r.choice or '-'}; options [{opts}]; rules said "
            f"{r.router_action} -> {name(r.router_target)}"
            + (f"; ERROR {r.error}" if r.error else "")
        )
    if decision.cues:
        out.append(f"cues: {', '.join(decision.cues)}")
    if decision.background is not None and decision.best_rel is not None:
        out.append(
            f"background similarity: {decision.background:.3f}   best rel: {decision.best_rel:.3f}"
        )
    if decision.gates:
        out.append("checks:")
        for g in decision.gates:
            out.append(
                f"  {g.name:22s} {g.value:6.3f} vs {g.threshold:6.3f}  "
                f"{'pass' if g.passed else 'fail'}  slack {g.slack:.3f}  {g.note}"
            )
    if decision.threads:
        out.append("threads:            best_rel  prior  score")
        for t in decision.threads:
            out.append(
                f"  {name(t.root_id):16s}   {t.best_rel:6.3f} {t.prior:6.3f} {t.score:6.3f}"
                + ("" if t.eligible else "  (below attach)")
            )
    if decision.candidates:
        out.append("candidates:   rel    sem    lex   matched")
        for c in decision.candidates:
            out.append(
                f"  {name(c.node_id):8s} {c.rel:6.3f} {c.sem:6.3f} {c.lex:6.3f}   "
                + " ".join(c.matched)
            )
    return "\n".join(out)
