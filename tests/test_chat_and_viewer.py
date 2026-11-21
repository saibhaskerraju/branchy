from __future__ import annotations

import json

from branchy import BranchingChat
from branchy.viewer.core import ViewerCore

CANNED = {
    "How do I deploy a Flask app with Docker?":
        "Write a Dockerfile using the python:3.12-slim base image, then docker build and run.",
    "How do I add a healthcheck to the container?":
        "Add a HEALTHCHECK line to the Dockerfile that curls /health.",
    "Give me a vegan pasta sauce recipe":
        "Creamy tomato-basil: onion, garlic, tomatoes, cashews, basil.",
}

MODEL = "openai/fake-model"  # litellm.completion is replaced by the fake_llm fixture


def test_history_is_the_path_to_root(tmp_path, fake_llm):
    fake_llm.replies = CANNED
    with BranchingChat(model=MODEL, store=tmp_path / "b.sqlite3") as chat:
        chat.add_message("How do I deploy a Flask app with Docker?")
        chat.add_message("How do I add a healthcheck to the container?")
        chat.add_message("Give me a vegan pasta sauce recipe")
    h = fake_llm.histories()
    assert h[0] == []
    assert [m["role"] for m in h[1]] == ["user", "assistant"]
    assert h[2] == []  # new topic: no history
    assert fake_llm.judge_calls == []  # all FIRM: routing never called the model


def test_model_is_required():
    import pytest

    with pytest.raises(ValueError):
        BranchingChat(model="", store=None)


def test_reload_from_store_routes_identically(tmp_path, fake_llm):
    path = tmp_path / "b.sqlite3"
    fake_llm.replies = CANNED
    with BranchingChat(model=MODEL, store=path, tree_id="t") as chat:
        for m in CANNED:
            chat.add_message(m)
        live = chat.router.route(chat.tree, "Should the healthcheck use curl or wget?")
    with BranchingChat(model=MODEL, store=path, tree_id="t") as chat:
        assert len(chat.tree.messages) == 6
        again = chat.router.route(chat.tree, "Should the healthcheck use curl or wget?")
    assert again.model_dump() == live.model_dump()


def test_reply_to_and_new_topic_through_chat(tmp_path, fake_llm):
    with BranchingChat(model=MODEL, store=None) as chat:
        u1 = chat.add_message("How do I deploy a Flask app with Docker?")
        a1 = chat.tree.messages[u1].child_message_ids[0]
        chat.add_message("How do I add a healthcheck to the container?")
        u3 = chat.add_message("and the port?", reply_to=a1)
        assert chat.last_decision.action == "FORK"
        assert chat.tree.messages[u3].parent_message_id == a1
        u4 = chat.add_message("How do I add a healthcheck to the container?", new_topic=True)
        assert chat.tree.messages[u4].parent_message_id is None


def test_viewer_serves_the_conversation(tmp_path, fake_llm):
    fake_llm.replies = CANNED
    with BranchingChat(model=MODEL, store=tmp_path / "b.sqlite3") as chat:
        for m in CANNED:
            chat.add_message(m)
        core = ViewerCore(chat._snapshot_provider())
        status, _, page = core.handle("GET", "/")
        assert status == 200 and b"branchy tree" in page
        status, _, body = core.handle("GET", "/api/tree")
        snap = json.loads(body)
        nodes = snap["trees"][0]["nodes"]
        assert len(nodes) == 6
        routed = [n["routing"] for n in nodes if n["node_type"] == "user"]
        assert [r["action"] for r in routed] == ["NEW", "CONTINUE", "NEW"]
        status, _, detail = core.handle("GET", f"/api/node/{nodes[2]['message_id']}")
        assert status == 200
        assert "How do I deploy" in json.loads(detail)["history"]


def test_readable_views(fake_llm):
    fake_llm.replies = CANNED
    with BranchingChat(model=MODEL, store=None) as chat:
        assert chat.explain() == "no routing yet"
        chat.add_message("How do I deploy a Flask app with Docker?")
        chat.add_message("How do I add a healthcheck to the container?")
        assert chat.last_reply == CANNED["How do I add a healthcheck to the container?"]
        assert chat.describe() == "↳ continue under A1"
        a1 = chat.message_id("a1")
        chat.add_message("and the port?", reply_to=a1)
        assert chat.describe() == "⑂ fork under A1 (sister of U2) [explicit]"
        tree = chat.show_tree()
        assert tree.splitlines()[0] == "(conversation)"
        assert "U3 and the port? ⑂" in tree and "← active" in tree
        assert chat.show_subtree(chat.message_id("U2")).startswith("U2 ")
        assert "decided_by=explicit" in chat.explain()
        assert chat.message_id("A9") is None


def test_refresh_stickers_matches_build_sticker():
    """The one-pass refresh gives exactly the per-topic reference result."""
    from pathlib import Path

    from branchy import ConversationTree, ThreadRouter
    from branchy.router.sticker import build_sticker

    router = ThreadRouter()
    tree = ConversationTree()
    scen = Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "handwritten.json"
    for conv in json.loads(scen.read_text("utf-8"))[:5]:
        for turn in conv["turns"]:
            d = router.route(tree, turn["user"])
            tree.add_message(turn["user"], turn["assistant"],
                             parent_message_id=d.target_message_id)
            tree.refresh_stickers()
    roots = tree.get_root_messages()
    assert len(roots) >= 5
    for r in roots:
        assert tree.stickers[r.message_id].text == build_sticker(tree, r.message_id)


def test_viewer_skips_snapshot_when_version_unchanged(tmp_path, fake_llm):
    with BranchingChat(model=MODEL, store=tmp_path / "b.sqlite3") as chat:
        chat.add_message("How do I deploy a Flask app with Docker?")
        loads = []
        provider = chat._snapshot_provider()

        def counting():
            loads.append(1)
            return provider()

        core = ViewerCore(counting, version_provider=chat._version_provider())
        status, headers, _ = core.handle("GET", "/api/tree")
        assert status == 200 and len(loads) == 1
        etag = headers["etag"]
        status, headers2, body = core.handle("GET", "/api/tree", etag)
        assert status == 304 and body == b"" and len(loads) == 1  # nothing loaded
        assert headers2["etag"] == etag
        chat.add_message("Give me a vegan pasta sauce recipe")
        status, headers3, _ = core.handle("GET", "/api/tree", etag)
        assert status == 200 and len(loads) == 2 and headers3["etag"] != etag
