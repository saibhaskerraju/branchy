from __future__ import annotations

import pytest

from branchy import RouterConfig, ThreadRouter

from conftest import Script

DOCKER_1 = (
    "How do I deploy a Flask app with Docker?",
    "Write a Dockerfile using the python:3.12-slim base image, install the "
    "requirements, run gunicorn. Then docker build and docker run the container.",
)
DOCKER_2 = (
    "How do I add a healthcheck to the container?",
    "Add a HEALTHCHECK line to the Dockerfile that curls the /health endpoint "
    "every 30 seconds.",
)
TOMATO = (
    "What pot size should I use for tomatoes on a balcony?",
    "Use a 20-litre pot with drainage holes for each tomato plant, in full sun.",
)
BASE_IMAGE = (
    "Going back to the base image, is alpine better than slim?",
    "Alpine is smaller but uses musl, so many Python wheels must compile from "
    "source; slim is usually the safer choice.",
)


def test_walkthrough_new_continue_new_fork(script: Script):
    """The 4-turn example from the design discussion."""
    d1 = script.say(*DOCKER_1)
    assert (d1.action, d1.target_message_id) == ("NEW", None)

    d2 = script.say(*DOCKER_2)
    assert d2.action == "CONTINUE"
    assert d2.target_message_id == script.A(1)

    d3 = script.say(*TOMATO)  # shares only common words with the Docker thread
    assert d3.action == "NEW"

    d4 = script.say(*BASE_IMAGE)
    assert "RETURN" in d4.cues
    assert d4.action == "FORK"
    assert d4.target_message_id == script.A(1)
    # U4 is a sister of U2: both are children of A1
    assert script.tree.messages[script.A(1)].child_message_ids == [script.U(2), script.U(4)]
    # the history the LLM gets for turn 4 is U1, A1 only
    path = [n.message_id for n in script.tree.get_path_to_root(d4.target_message_id)]
    assert path == [script.U(1), script.A(1)]


def test_fanout_is_unbounded(script: Script):
    """Every return to the same answer adds another sister."""
    script.say(*DOCKER_1)
    script.say(*DOCKER_2)
    assert script.say(*BASE_IMAGE).action == "FORK"  # 2nd child of A1
    d = script.say(
        "Going back to the Dockerfile basics, how many gunicorn workers should it run?",
        "Start with 2 x CPU cores + 1 workers.",
    )
    assert d.action == "FORK"
    assert d.target_message_id == script.A(1)
    assert len(script.tree.messages[script.A(1)].child_message_ids) == 3


def test_return_cue_does_not_fork_without_distinct_evidence(script: Script):
    """'Going back' to content the current branch also covers: continuing
    keeps the earlier answer in context, so no fork."""
    script.say(*DOCKER_1)
    script.say(*DOCKER_2)
    script.say(*BASE_IMAGE)  # the alpine branch is about base images too
    d = script.say(
        "Going back to the Dockerfile basics: should I pin the python base image version?",
        "Yes, pin python:3.12.4-slim.",
    )
    assert d.action == "CONTINUE"
    g1 = next(g for g in d.gates if g.name == "fork_G1_distinct")
    assert not g1.passed


def test_meaning_only_weak_match_is_never_firm(script: Script):
    """Known limit of a static model: an unrelated question can get a small
    semantic lift ("size" ~ "slim image"). It may attach, but only TENTATIVE."""
    script.say("How do I deploy a Flask app with Docker?",
               "Write a Dockerfile using the python:3.12-slim base image, then docker build and run.")
    d = script.say("What pot size should I use for tomatoes on a balcony?", "x")
    assert d.action == "NEW" or d.confidence == "TENTATIVE"


def test_stemmed_continuation_is_not_a_new_topic(script: Script):
    script.say(*DOCKER_1)
    d = script.say(
        "What about multi-stage builds to shrink the image?",
        "Use a builder stage, then copy the installed packages into a slim stage.",
    )
    assert d.action == "CONTINUE"
    assert d.target_message_id == script.A(1)


def test_reference_cue_never_starts_a_new_topic(script: Script):
    script.say(*TOMATO)
    d = script.say("How often should I water them?", "Daily in hot weather.")
    assert "ANAPHORA" in d.cues
    assert d.action == "CONTINUE"
    d = script.say("Thanks!", "You're welcome.")
    assert "FRAGMENT" in d.cues
    assert d.action == "CONTINUE"


def test_reference_cue_goes_to_the_matching_thread(script: Script):
    """'it' + content that clearly belongs to an older thread."""
    script.say(*DOCKER_1)
    script.say(*TOMATO)
    d = script.say("Should the Dockerfile run it with gunicorn workers?",
                   "Yes, use 2-4 workers per core.")
    assert d.action == "SWITCH"
    assert d.target_message_id == script.A(1)


def test_new_topic_cue(script: Script):
    script.say(*DOCKER_1)
    d = script.say("Unrelated question: what is a good index fund for retirement?",
                   "A low-cost total market index fund is a common choice.")
    assert "NEW_TOPIC" in d.cues
    assert d.action == "NEW"


def test_returning_to_a_thread_switches_to_its_leaf(script: Script):
    script.say(*DOCKER_1)
    script.say(*DOCKER_2)
    script.say(*TOMATO)
    d = script.say("For the Docker healthcheck, what interval should the curl use?",
                   "30 seconds with a 3 second timeout is typical.")
    assert d.action == "SWITCH"
    assert d.target_message_id == script.A(2)


def test_switch_back_to_another_branch(script: Script):
    script.say(*DOCKER_1)
    script.say(*DOCKER_2)
    script.say(*BASE_IMAGE)  # fork: the alpine branch is now active
    d = script.say("Going back to the healthcheck, how often should it curl the endpoint?",
                   "Every 30 seconds.")
    assert d.action == "SWITCH"
    assert d.target_message_id == script.A(2)


def test_explicit_reply_to_and_new_topic(script: Script):
    script.say(*DOCKER_1)
    script.say(*DOCKER_2)
    d = script.say("anything", "ok", reply_to=script.A(1))
    assert (d.action, d.decided_by, d.confidence) == ("FORK", "explicit", "FIRM")
    d = script.say("How do I add a healthcheck?", "x", new_topic=True)
    assert (d.action, d.target_message_id, d.decided_by) == ("NEW", None, "explicit")
    with pytest.raises(ValueError):
        script.router.route(script.tree, "x", reply_to=script.U(1))


def test_decisions_are_deterministic(router):
    turns = [DOCKER_1, DOCKER_2, TOMATO, BASE_IMAGE]
    a, b = Script(router), Script(ThreadRouter())  # fresh router: no caches

    def canon(s: Script):
        index = {mid: f"U{i+1}" for i, mid in enumerate(s.users)}
        index.update({mid: f"A{i+1}" for i, mid in enumerate(s.assistants)})
        out = []
        for d in s.decisions:
            dump = d.model_dump()
            dump["target_message_id"] = index.get(d.target_message_id)
            dump["thread_id"] = index.get(d.thread_id)
            for c in dump["candidates"]:
                c["node_id"], c["root_id"] = index[c["node_id"]], index[c["root_id"]]
            for t in dump["threads"]:
                t["root_id"] = index[t["root_id"]]
            for g in dump["gates"]:
                g["note"] = ""
            out.append(dump)
        return out

    for t in turns:
        a.say(*t)
        b.say(*t)
    assert canon(a) == canon(b)


def test_candidate_scores_are_bounded_and_rounded(script: Script):
    script.say(*DOCKER_1)
    d = script.say(*DOCKER_2)
    for c in d.candidates:
        assert 0.0 <= c.rel <= 1.0 and 0.0 <= c.sem <= 1.0 and 0.0 <= c.lex <= 1.0
        assert c.rel == round(c.rel, 6)


def test_uncertain_band_is_tentative():
    """Squeeze the attach band so a weak match lands in it."""
    cfg = RouterConfig(tau_new=0.0, tau_attach=1.0)
    s = Script(ThreadRouter(cfg))
    s.say(*DOCKER_1)
    d = s.say(*DOCKER_2)
    assert d.confidence == "TENTATIVE"
    assert d.action == "CONTINUE"


# --- regressions from a live session (eval/scenarios/live.json) -------------

INTRO = (
    "who are you",
    "I'm an AI assistant. I can help answer questions, explain things, "
    "brainstorm, write, code, and more.",
)
PODS = (
    "explain kubernetes pods",
    "A pod is the smallest deployable unit in Kubernetes. It wraps one or more "
    "containers that share a network namespace and storage volumes.",
)


def test_reference_does_not_switch_to_weak_rival(script: Script):
    """'in that' points at the answer just given. A generic intro answer that
    merely leads the active thread (without attach-level evidence) must not
    take the reference."""
    script.say(*INTRO)
    script.say("what is docker",
               "Docker packages an application and its dependencies into a "
               "container that runs the same everywhere.")
    d = script.say("can we use fastapi in that", "Yes, run it with uvicorn in the image.")
    assert d.target_message_id == script.A(2)
    assert d.tentative
    switch = [g for g in d.gates if g.name == "thread_switch_evidence"]
    assert all(not g.passed for g in switch)


def test_rare_shared_term_returns_to_its_thread(script: Script):
    """A rare name shared with a non-active thread attaches there even in the
    uncertain band, instead of starting a new topic."""
    script.say("How do I create a migration with alembic?",
               "Run alembic revision --autogenerate, then alembic upgrade head.")
    script.say("Suggest a weekend trip near Lisbon",
               "Sintra: palaces, forest trails and Cabo da Roca nearby.")
    d = script.say("can alembic handle multiple databases", "Yes, use the multidb template.")
    assert d.target_message_id == script.A(1)
    assert d.action == "SWITCH"


def test_common_shared_word_still_new(script: Script):
    """The rare-term rule must not pull a new topic in on a common word."""
    script.say(*DOCKER_1)
    script.say(*DOCKER_2)
    assert script.say(*TOMATO).action == "NEW"


def test_short_request_is_not_a_fragment(script: Script):
    """'i want to understand X' is a request, not a fragment that must attach."""
    script.say(*INTRO)
    d = script.say("i want to understand docker in brief", "Docker packages apps.")
    assert "FRAGMENT" not in d.cues
    assert d.action == "NEW"


BOOKING = (
    "I need to book a dentist appointment for next week.",
    "Dr. Patel has openings on Tuesday at 10:00 and Thursday at 15:30. "
    "Would you like me to book Tuesday at 10:00?",
)


def test_short_answer_to_assistant_question_is_firm(script: Script):
    """'yes' right after the assistant asked something: continue, FIRM (no LLM)."""
    script.say(*BOOKING)
    for reply in ("yes", "Yes, can you also email me the invoice?"):
        d = script.say(reply, "Done. Do you want a reminder the day before?")
        assert d.action == "CONTINUE", reply
        assert d.confidence == "FIRM", reply
        assert any(g.name == "answers_question" for g in d.gates), reply


def test_bare_yes_without_a_question_stays_tentative(script: Script):
    script.say(*TOMATO)
    d = script.say("yes", "Great, happy growing.")
    assert d.action == "CONTINUE"
    assert d.confidence == "TENTATIVE"


def test_personal_pronoun_follow_up_is_not_new(script: Script):
    script.say("Who directed the film Dunkirk?",
               "Christopher Nolan wrote and directed Dunkirk (2017).")
    d = script.say("where was he born?", "Nolan was born in London in 1970.")
    assert "ANAPHORA" in d.cues
    assert d.action == "CONTINUE"


def test_return_cue_without_matching_words_is_not_new(script: Script):
    script.say(*DOCKER_1)
    d = script.say("can you describe the first one?", "The first step is the Dockerfile.")
    assert "RETURN" in d.cues
    assert d.action != "NEW"
