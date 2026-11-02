"""Use the deterministic router without BranchingChat.

What it shows: ThreadRouter + ConversationTree on their own, the full
RoutingDecision (candidates, evidence, dominated roots), a RouterConfig
override, and a determinism check.
Run:  python examples/router_only.py
Needs LLM: no.  Writes nothing to disk.
"""

from __future__ import annotations

from _common import SAMPLE_TURNS

from branchy import ConversationTree, RouterConfig, ThreadRouter
from branchy.router.lexical import DEFAULT_CUES


def build_tree(router: ThreadRouter) -> ConversationTree:
    tree = ConversationTree()
    for user, assistant in SAMPLE_TURNS:
        decision = router.route(tree, user)
        tree.add_message(user, assistant, parent_message_id=decision.target_message_id,
                         tentative=decision.action == "ATTACH_TENTATIVE",
                         routing=decision.model_dump())
        tree.refresh_stickers()  # keep stickers current; BranchingChat does this for you
    return tree


def show(decision) -> None:
    print(f"{decision.action}  target={decision.target_message_id and decision.target_message_id[:8]}"
          f"  margin={decision.margin}")
    print(f"  reason: {decision.reason}")
    for c in decision.candidates:
        print(f"  {c.node_id[:8]} {c.kind:4s} bm25={c.bm25:.2f} rec={c.recency:.2f} "
              f"s1={c.s1:.2f} evidence={c.evidence}"
              + ("  (dominated root)" if c.node_id in decision.dominated_root_ids else ""))


def main() -> None:
    router = ThreadRouter()  # defaults: k=3, recency half-life 3 turns
    tree = build_tree(router)
    print(tree.render_tree(), "\n")

    print("route 'Should I use a GIN index for JSONB?':")
    show(router.route(tree, "Should I use a GIN index for JSONB?"))

    print("\nroute 'and the sauce?' (fragment):")
    show(router.route(tree, "and the sauce?"))

    # Config override: wider shortlist and an extra referential cue word.
    custom = ThreadRouter(config=RouterConfig(k=5, fragment_cues=DEFAULT_CUES | {"ditto"}))
    print("\nroute 'ditto for the salt' with custom config:")
    show(custom.route(tree, "ditto for the salt"))

    # Determinism: identical inputs -> identical decisions.
    a = router.route(tree, "Should I use a GIN index for JSONB?").model_dump()
    b = router.route(tree, "Should I use a GIN index for JSONB?").model_dump()
    print("\ndeterministic:", a == b)


if __name__ == "__main__":
    main()
