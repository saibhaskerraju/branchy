from __future__ import annotations

import pytest

from branchy.router.embed import Vec, cosine, load_router_data
from branchy.router.porter import stem
from branchy.router.text import detect_cues, terms


@pytest.mark.parametrize(
    "word,expected",
    [
        ("caresses", "caress"), ("ponies", "poni"), ("agreed", "agre"),
        ("motoring", "motor"), ("hopping", "hop"), ("happy", "happi"),
        ("relational", "relat"), ("generalizations", "gener"),
        ("oscillators", "oscil"), ("adjustable", "adjust"), ("builds", "build"),
        ("containers", "contain"), ("controll", "control"),
    ],
)
def test_porter_reference_cases(word, expected):
    assert stem(word) == expected


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Going back to the base image, is alpine better?", {"RETURN"}),
        ("btw what's a good index fund?", {"NEW_TOPIC"}),
        ("How often should I water them?", {"ANAPHORA"}),
        ("Thanks!", {"FRAGMENT"}),
        ("make it shorter", {"ANAPHORA", "FOLLOW_UP", "FRAGMENT"}),
        ("Give an example in Python", {"FOLLOW_UP"}),
        ("How long for a soft one?", {"ANAPHORA"}),
        ("What is one good book about history?", set()),
        ("What is Kubernetes?", set()),
        # long message: "that" is a relative pronoun, not a back-reference
        ("Write a Python script that pings my servers every minute", set()),
        ("where was he born?", {"ANAPHORA"}),
        ("which actor played the former?", {"ANAPHORA"}),
        ("yes", {"FRAGMENT", "AFFIRMATION"}),
        ("Yes, can you also email me the invoice?", {"AFFIRMATION"}),
        ("sure, go ahead", {"FRAGMENT", "AFFIRMATION"}),
    ],
)
def test_cues(message, expected):
    assert set(detect_cues(message).names()) == expected


def test_terms_are_stemmed_content_words():
    assert terms("What about multi-stage builds to shrink the image?") == [
        "multi", "stage", "build", "shrink", "imag",
    ]


def test_embeddings_are_exact_integers():
    data = load_router_data()
    v = data.embedder.embed("Deploy a Flask app with Docker")
    assert len(v.values) == data.embedder.dim
    assert all(isinstance(x, int) for x in v.values)
    assert v == data.embedder.embed("Deploy a Flask app with Docker")
    assert cosine(v, v) == pytest.approx(1.0)
    assert cosine(v, Vec.of([0] * data.embedder.dim)) == 0.0


def test_idf_prefers_rare_words():
    data = load_router_data()
    assert data.idf("kubernet") > data.idf("good")
    assert data.idf("zzzunknownzzz") == data.unknown_idf_milli
