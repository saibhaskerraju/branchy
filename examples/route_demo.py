"""Route demo: canned-reply scenarios; no LLM call is ever made.

(A) scenarios 1+2, lexical
(D) scenario 3, lexical
(E) determinism: scenario 2, lexical, twice

--viewer starts the HTML tree viewer mid-run; --viewer-seconds N auto-exits.
"""

from __future__ import annotations

import json
import os
import sys
import time

from branchy import BranchingChat, RoutingDecision

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCENARIO_1 = [
    (
        "How do I boil an egg?",
        "Put the egg in boiling water for about 10 minutes.",
    ),
    (
        "Can you tell me a joke?",
        "Why did the egg hide? It was a little chicken.",
    ),
    (
        "How long for a soft one?",
        "About 6 minutes for a soft yolk.",
    ),
]

SCENARIO_2 = [
    (
        "How do I build an index in Postgres?",
        "Use CREATE INDEX idx ON table(col); btree by default.",
    ),
    (
        "Give me a vegan pasta sauce recipe",
        "Creamy tomato-basil: onion, garlic, tomatoes, cashews, basil.",
    ),
    (
        "What if I need it to not lock the table?",
        "Use CREATE INDEX CONCURRENTLY; it cannot run inside a transaction.",
    ),
    (
        "I don't like onion, can you change it?",
        "No-onion version: extra garlic, fennel seed, tomatoes, cashews.",
    ),
    (
        "How much salt?",
        "Half a teaspoon when adding tomatoes, adjust after simmering.",
    ),
    (
        "Can that run inside a migration transaction?",
        "No — CONCURRENTLY cannot run in a transaction block; run it outside the migration.",
    ),
]

SCENARIO_3 = [
    (
        "How do I build an index in Postgres?",
        "Use CREATE INDEX idx ON table(col).",
    ),
    (
        "Give me a vegan pasta sauce recipe",
        "Creamy tomato-basil: onion, garlic, tomatoes, cashews.",
    ),
    (
        "Make the index build without blocking writers",
        "Use CREATE INDEX CONCURRENTLY.",
    ),
    (
        "Poutine recipe",
        "Fries, cheese curds, gravy.",
    ),
    (
        "Swap the cashews for something nut-free",
        "Use silken tofu or white beans.",
    ),
    (
        "Thanks!",
        "You're welcome.",
    ),
]


def _fmt(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:6.3f}"


def _chat() -> BranchingChat:
    # the responder is never invoked: every turn passes assistant_message=
    # store=None keeps the demo side-effect free (no .branchy/ writes)
    return BranchingChat(responder=lambda history, message: "", store=None)


def run_scenario(
    name: str,
    turns: list[tuple[str, str]],
    show_tree: bool = True,
) -> list[dict]:
    chat = _chat()
    print(f"\n=== {name} ===")
    records: list[dict] = []
    for index, (user, assistant) in enumerate(turns, start=1):
        chat.add_message(user, assistant_message=assistant)
        decision: RoutingDecision = chat.last_decision
        print(
            f"turn {index}: {decision.action} target={decision.target_message_id} "
            f"margin={_fmt(decision.margin)}"
        )
        print(f"  reason: {decision.reason}")
        for cand in decision.candidates:
            print(
                f"    cand {cand.node_id[:8]} kind={cand.kind} ev={cand.evidence} "
                f"bm25={_fmt(cand.bm25)} rec={_fmt(cand.recency)} s1={_fmt(cand.s1)}"
            )
        records.append(decision.model_dump())
    if show_tree:
        print(chat.show_tree())
    return records


def _canonical(records: list[dict]) -> str:
    """Determinism signature across two fresh trees. Message ids are random
    uuids, so node↔tower identity is compared structurally instead: the target
    is expressed as its tower's index in root (created_seq) order. Candidate
    order is kept as emitted — the router's tie-breaks are created_seq-based,
    so ordering is deterministic too."""
    root_index: dict[str, int] = {}

    def tower(node_id: str | None) -> int | None:
        if node_id is None:
            return None
        if node_id not in root_index:
            root_index[node_id] = len(root_index)
        return root_index[node_id]

    out = []
    for record in records:
        target_root = None
        for cand in record["candidates"]:
            if cand["node_id"] == record["target_message_id"]:
                target_root = cand["root_id"]
        score_tuples = [
            (
                c["kind"],
                c["bm25"],
                c["recency"],
                c["s1"],
                c["evidence"],
            )
            for c in record["candidates"]
        ]
        out.append(
            {
                "action": record["action"],
                "target_tower": tower(target_root),
                "margin": record["margin"],
                "dominated_towers": [
                    tower(
                        next(
                            c["root_id"]
                            for c in record["candidates"]
                            if c["node_id"] == nid
                        )
                    )
                    for nid in record["dominated_root_ids"]
                ],
                "shortlisted_towers": [
                    tower(
                        next(
                            c["root_id"]
                            for c in record["candidates"]
                            if c["node_id"] == nid
                        )
                    )
                    for nid in record["shortlisted_ids"]
                ],
                "candidate_scores": score_tuples,
            }
        )
    return json.dumps(out, sort_keys=True)


def _fragment_self_check() -> None:
    from branchy.router.lexical import is_fragment

    expectations = [
        ("Can you tell me a joke?", False),
        ("How much salt?", True),
        ("Why?", True),
        ("Thanks!", True),
        ("Convert 5 miles to km", False),
        ("What's the capital of France?", False),
        ("How long for a soft one?", True),
        ("Make it shorter", True),
    ]
    for message, expected in expectations:
        actual = is_fragment(message)
        assert actual == expected, f"{message!r}: expected {expected}, got {actual}"
    print("fragment self-check OK")


VIEWER_EXTRA_TURNS = [
    ("Can I add a partial index?", "Yes — CREATE INDEX with a WHERE clause."),
    ("Should the sauce be blended?", "Blend half for a creamier texture."),
    ("What index type for JSONB?", "Use a GIN index on the jsonb column."),
    ("Thanks!", "You're welcome."),
]


def _viewer_demo(viewer_seconds: int | None) -> None:
    print("\n=== viewer demo (scenario 2 + live turns) ===")
    chat = _chat()
    for user, assistant in SCENARIO_2:
        chat.add_message(user, assistant_message=assistant)
    chat.start_viewer()
    print(f"viewer: {chat.viewer_url}")
    for user, assistant in VIEWER_EXTRA_TURNS:
        time.sleep(3)
        chat.add_message(user, assistant_message=assistant)
        print(f"added turn: {user!r} -> {chat.last_decision.action}")
    if viewer_seconds is not None:
        time.sleep(viewer_seconds)
    else:
        input("viewer running — press Enter to stop")
    out = os.path.join(os.path.dirname(__file__), "out", "route_demo_tree.html")
    chat.export_viewer_html(out)
    print(f"exported static viewer: {out}")
    chat.stop_viewer()


def main() -> None:
    with_viewer = "--viewer" in sys.argv
    viewer_seconds = None
    if "--viewer-seconds" in sys.argv:
        viewer_seconds = int(
            sys.argv[sys.argv.index("--viewer-seconds") + 1]
        )

    _fragment_self_check()

    # (A) lexical
    run_scenario("Scenario 1", SCENARIO_1)
    run_scenario("Scenario 2", SCENARIO_2)

    # (D) scenario 3
    run_scenario("Scenario 3", SCENARIO_3)

    # (E) determinism: scenario 2, lexical, twice
    first = run_scenario(
        "Scenario 2 determinism run A", SCENARIO_2, show_tree=False
    )
    second = run_scenario(
        "Scenario 2 determinism run B", SCENARIO_2, show_tree=False
    )
    if with_viewer:
        _viewer_demo(viewer_seconds)

    a = _canonical(first)
    b = _canonical(second)
    print(f"\ndeterminism check: {'IDENTICAL' if a == b else 'DIFFERS'}")
    if a != b:
        sys.exit(1)


if __name__ == "__main__":
    main()
