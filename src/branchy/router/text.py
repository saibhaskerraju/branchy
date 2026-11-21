"""Text analysis for routing: terms (stemmed content words) and cue flags.

English only. Everything here is pure and deterministic.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .porter import stem

_WORD_RE = re.compile(r"[a-z0-9_]+")
# "a soft one", "the other ones", "that one": "one" used as a pronoun
_ONE_PRONOUN_RE = re.compile(r"\b(?:a|an|the|that|this|which|another)\s+(?:[a-z]+\s+)?ones?\b")

STOPWORDS = frozenset(
    """
    a an the and or but if then else for to of in on at by with about as is are
    was were be been being do does did doing done have has had having i you he
    she it we they me him her us them my your yours his its our their this that
    these those there here what which who whom whose when where why how not no
    so too very can could will would shall should may might must just also than
    into from up down out over under again more most some any each all both few
    other such only own same s t d ll m re ve don doesn didn isn aren wasn
    weren won wouldn shouldn couldn now get got like want need please thanks
    thank ok okay yes yeah hi hello hey let lets im ive id youre dont cant one
    ones make made thing things way really much many well still even something
    anything everything know think tell say said btw going go use used
    using uses
    """.split()
)

# A message containing one of these words is asking for something, so a short
# message with one is not treated as a fragment ("What is Kubernetes?").
REQUEST_MARKERS = frozenset(
    """
    you me tell give show write explain list create convert define describe
    summarize summarise translate generate compare recommend suggest what who
    where when which how why can could should would is are do does
    want need understand learn know
    """.split()
)

# Words that point back at something already said.
ANAPHORA_WORDS = frozenset(
    "it its it's that this those these them they ones same another again "
    "he him his she her hers their theirs there here "
    "former latter aforesaid aforementioned above mentioned".split()
)

# A short message opening with one of these answers the assistant's question
# ("yes", "sure, go ahead", "no thanks", "ok, book it").
AFFIRMATION_OPENERS = (
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "no", "nope", "nah",
    "please do", "go ahead", "do it", "sounds good", "that works", "correct",
    "exactly", "absolutely", "of course", "definitely", "agreed",
    "confirm", "confirmed", "got it", "perfect",
)

# Phrases (matched on whole words) that signal going back to earlier content.
RETURN_PHRASES = (
    "go back to", "going back to", "get back to", "back to the", "back to what",
    "back to our", "back to my", "back to your", "coming back to", "return to",
    "returning to", "revisit", "earlier", "previously", "you mentioned",
    "you said", "as you said", "originally", "at the start", "in the beginning",
    "the first one", "as before",
)

# Phrases that ask for more of the previous answer without naming its topic
# ("Give an example in Python").
FOLLOW_UP_PHRASES = (
    "an example", "another example", "for example", "examples", "elaborate",
    "more detail", "more details", "in detail", "explain more", "explain further",
    "go on", "keep going", "tell me more", "expand on", "simpler", "shorter",
    "longer", "rewrite", "rephrase", "in python", "in javascript", "in typescript",
    "in java", "in rust", "in go", "in c", "in sql", "in bash", "step by step",
    "why is that", "how so", "what else",
)

# A short message opening with one of these continues the previous answer
# ("And how are mRNA ones different?", "What about Rust?").
FOLLOW_UP_OPENERS = (
    "and", "also", "but", "so", "then", "plus", "what about", "how about",
    "what if", "and what", "and how", "ok and", "okay and",
)

# Phrases that announce a new topic.
NEW_TOPIC_PHRASES = (
    "unrelated", "different question", "different topic", "new question",
    "new topic", "change of topic", "change the subject", "changing the subject",
    "switching gears", "switch gears", "by the way", "btw", "off topic",
    "another question", "random question", "something else", "on another note",
    "separate question", "totally different",
)


def words(text: str) -> list[str]:
    """Lowercase word tokens ([a-z0-9_]+), apostrophes removed first."""
    return _WORD_RE.findall(text.lower().replace("'", "").replace("’", ""))


def content_words(text: str) -> list[str]:
    """Words of length >= 2 that are not stopwords (unstemmed)."""
    return [w for w in words(text) if len(w) >= 2 and w not in STOPWORDS]


def strip_meta(text: str) -> str:
    """Remove RETURN and NEW_TOPIC phrases: they talk about the conversation
    ("going back to", "unrelated question"), not about its topic. Returns
    lowercase words joined by spaces."""
    joined = " " + " ".join(words(text)) + " "
    for phrase in _META_PHRASES:
        joined = joined.replace(f" {phrase} ", " ")
    return joined.strip()


def topic_words(text: str) -> list[str]:
    """Content words of the text with conversation meta-phrases removed."""
    return content_words(strip_meta(text))


def topic_text(text: str) -> str:
    """The text the router embeds: topic words only. Function words carry
    the style of a request ("how do I ... should I"), not its topic, and make
    unrelated how-to questions look alike."""
    return " ".join(topic_words(text))


def terms(text: str) -> list[str]:
    """Stemmed topic words, in order (duplicates kept)."""
    return [stem(w) for w in topic_words(text)]


def term_set(text: str) -> frozenset[str]:
    return frozenset(terms(text))


_META_PHRASES = tuple(sorted(set(RETURN_PHRASES + NEW_TOPIC_PHRASES), key=len, reverse=True))


def _has_phrase(joined: str, phrases: tuple[str, ...]) -> bool:
    return any(f" {p} " in joined for p in phrases)


@dataclass(frozen=True)
class Cues:
    """Cue flags of one message. See ADR 0002 for how each one is used."""

    return_: bool = False
    new_topic: bool = False
    anaphora: bool = False
    follow_up: bool = False
    fragment: bool = False
    affirmation: bool = False

    @property
    def refers_back(self) -> bool:
        """The message leans on earlier context: it must not start a topic."""
        return self.fragment or self.anaphora or self.follow_up or self.affirmation

    def names(self) -> list[str]:
        out = []
        if self.return_:
            out.append("RETURN")
        if self.new_topic:
            out.append("NEW_TOPIC")
        if self.anaphora:
            out.append("ANAPHORA")
        if self.follow_up:
            out.append("FOLLOW_UP")
        if self.fragment:
            out.append("FRAGMENT")
        if self.affirmation:
            out.append("AFFIRMATION")
        return out


def detect_cues(
    text: str,
    *,
    fragment_max_terms: int = 3,
    anaphora_max_terms: int = 6,
    return_phrases: tuple[str, ...] = RETURN_PHRASES,
    new_topic_phrases: tuple[str, ...] = NEW_TOPIC_PHRASES,
    follow_up_phrases: tuple[str, ...] = FOLLOW_UP_PHRASES,
    anaphora_words: frozenset = ANAPHORA_WORDS,
    request_markers: frozenset = REQUEST_MARKERS,
) -> Cues:
    """Cue flags for a message.

    - RETURN / NEW_TOPIC: one of the phrases occurs (whole words).
    - ANAPHORA: a pointing-back word occurs and the message is short
      (<= anaphora_max_terms content words) — in a long message "that" is
      usually a relative pronoun ("a script that pings servers").
    - FOLLOW_UP: a phrase asking for more of the last answer, or an opening
      conjunction ("And ...", "What about ..."), in a short message (same
      length limit as ANAPHORA).
    - FRAGMENT: <= fragment_max_terms content words and no request marker.
    - AFFIRMATION: opens with a yes/no-style answer ("yes", "sure, go ahead",
      "no thanks") in a short message (same length limit as ANAPHORA).
    """
    ws = words(text)
    joined = " " + " ".join(ws) + " "
    n_content = len([w for w in ws if len(w) >= 2 and w not in STOPWORDS])
    raw = text.lower()
    has_anaphor = (
        any(w in anaphora_words for w in ws)
        or "it's" in raw
        or _ONE_PRONOUN_RE.search(" ".join(ws)) is not None
    )
    return Cues(
        return_=_has_phrase(joined, return_phrases),
        new_topic=_has_phrase(joined, new_topic_phrases),
        anaphora=has_anaphor and n_content <= anaphora_max_terms,
        follow_up=(
            _has_phrase(joined, follow_up_phrases)
            or any(joined.startswith(f" {o} ") for o in FOLLOW_UP_OPENERS)
        )
        and n_content <= anaphora_max_terms,
        fragment=n_content <= fragment_max_terms
        and not any(w in request_markers for w in ws),
        affirmation=any(joined.startswith(f" {o} ") for o in AFFIRMATION_OPENERS)
        and n_content <= anaphora_max_terms,
    )


def asks_question(reply: str, tail_chars: int = 300) -> bool:
    """The reply ends by asking the user something: a '?' near its end
    ("... Would you like me to book Tuesday?")."""
    return "?" in reply.rstrip()[-tail_chars:]


def top_keywords(
    thread_tokens: list[str],
    other_threads_tokens: list[list[str]],
    k: int = 8,
) -> list[str]:
    """TF in this thread x 1/(1 + threads containing term); ties by (-score, token)."""
    tf = Counter(thread_tokens)
    other_sets = [set(doc) for doc in other_threads_tokens]
    scored = []
    for term, count in tf.items():
        containing = sum(1 for doc in other_sets if term in doc)
        scored.append((count / (1 + containing), term))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [term for _, term in scored[:k]]
