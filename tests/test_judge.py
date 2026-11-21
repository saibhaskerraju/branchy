from __future__ import annotations

from branchy import BranchingChat
from branchy.router import RoutingJudge

from conftest import Script


def fake(answer):
    """A complete() that records prompts and returns a fixed answer (or raises)."""
    calls = []

    def complete(messages):
        calls.append(messages)
        if isinstance(answer, Exception):
            raise answer
        return answer

    complete.calls = calls
    return complete


def test_firm_decisions_are_not_reviewed(script: Script):
    script.say("How do I build an index in Postgres?", "Use CREATE INDEX.")
    d = script.router.route(script.tree, "Give me a vegan pasta sauce recipe")
    assert not d.tentative
    complete = fake("A")
    out = RoutingJudge(complete).review(script.tree, "x", d, script.router)
    assert out is d and not complete.calls


def test_override_to_new_topic(script: Script):
    script.say("who are you", "I'm an AI assistant. I can help with code and writing.")
    msg = "can we use fastapi in that"  # reference cue, weak evidence -> TENTATIVE
    d = script.router.route(script.tree, msg)
    assert d.tentative and d.target_message_id == script.A(1)
    out = RoutingJudge(fake("N")).review(script.tree, msg, d, script.router)
    assert out.target_message_id is None and out.action == "NEW"
    assert out.decided_by == "llm"
    assert out.llm.router_target == script.A(1) and not out.llm.agreed


def test_options_most_recent_first_then_new(script: Script):
    script.say("How do I build an index in Postgres?", "Use CREATE INDEX.")
    script.say("Give me a vegan pasta sauce recipe", "Tomato-basil with cashews.")
    d = script.router.route(script.tree, "how long does that take")
    judge = RoutingJudge(fake("A"))
    opts = judge.options(script.tree, d, script.router)
    assert opts[0] == script.A(2) and opts[-1] is None and script.A(1) in opts
    prompt = judge.prompt(script.tree, "how long does that take", opts)[1]["content"]
    assert "[A] (most recent exchange)" in prompt and "[N]" in prompt


def test_agreement_keeps_rules_decision(script: Script):
    script.say("How do I build an index in Postgres?", "Use CREATE INDEX.")
    script.say("Give me a vegan pasta sauce recipe", "Tomato-basil with cashews.")
    msg = "how long does that take"
    d = script.router.route(script.tree, msg)
    assert d.tentative
    judge = RoutingJudge(fake("A"))
    letter = "ABCD"[judge.options(script.tree, d, script.router).index(d.target_message_id)]
    out = RoutingJudge(fake(letter)).review(script.tree, msg, d, script.router)
    assert out.target_message_id == d.target_message_id
    assert out.decided_by == "rules" and out.llm.agreed


def test_failures_keep_rules_decision(script: Script):
    script.say("who are you", "I'm an AI assistant.")
    msg = "can we use fastapi in that"
    d = script.router.route(script.tree, msg)
    for answer in (TimeoutError("slow"), "I am not sure", "Z"):
        out = RoutingJudge(fake(answer)).review(script.tree, msg, d, script.router)
        assert out.target_message_id == d.target_message_id
        assert out.decided_by == "rules" and out.llm.error


def test_kinds_filter(script: Script):
    script.say("who are you", "I'm an AI assistant.")
    msg = "can we use fastapi in that"
    d = script.router.route(script.tree, msg)
    assert d.uncertainty == "reference"
    complete = fake("N")
    out = RoutingJudge(complete, kinds=["band"]).review(script.tree, msg, d, script.router)
    assert out is d and not complete.calls


def test_chat_reviews_only_tentative_decisions(fake_llm):
    fake_llm.judge_answer = "N"
    chat = BranchingChat(model="openai/fake-model", store=None)
    chat.add_message("who are you")                 # empty tree: FIRM
    assert fake_llm.judge_calls == [] and chat.last_decision.llm is None
    chat.add_message("can we use fastapi in that")  # weak reference: TENTATIVE
    assert len(fake_llm.judge_calls) == 1
    assert chat.last_decision.action == "NEW" and chat.last_decision.decided_by == "llm"
    node = chat.tree.messages[chat.last_message_id]
    assert node.parent_message_id is None
    assert node.routing["llm"]["router_action"] == "CONTINUE"
    assert "llm overrode rules" in chat.describe()
    # the reply was written after routing, with the LLM's placement (no history)
    assert fake_llm.histories()[-1] == []


def test_chat_keeps_rules_when_review_fails(fake_llm):
    fake_llm.judge_answer = "no idea"
    chat = BranchingChat(model="openai/fake-model", store=None)
    chat.add_message("who are you")
    chat.add_message("can we use fastapi in that")
    d = chat.last_decision
    assert d.decided_by == "rules" and d.action == "CONTINUE" and d.llm.error
