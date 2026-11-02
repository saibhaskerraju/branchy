from __future__ import annotations

import math
import re
from collections import Counter

_STOPWORDS = frozenset(
    "a an the and or but if then else for to of in on at by with about as is are was "
    "were be been being do does did doing have has had having i you he she it we they "
    "me him her us them my your his its our their this that these those there here "
    "what which who whom when where why how not no so too very can will just".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

DEFAULT_CUES = frozenset(
    "it its that this one ones other others same again also more instead them "
    "those these another else too".split()
)

DEFAULT_REQUEST_MARKERS = frozenset(
    "you me tell give show write explain list create convert define describe "
    "summarize summarise translate generate what who where when which".split()
)


def has_cue(message: str, cues: frozenset = DEFAULT_CUES) -> bool:
    """True when a raw token of the message is a referential cue word."""
    raw = _TOKEN_RE.findall(message.lower())
    return any(t in cues for t in raw)


def is_fragment(
    message: str,
    cues: frozenset = DEFAULT_CUES,
    markers: frozenset = DEFAULT_REQUEST_MARKERS,
) -> bool:
    """Heuristic. A message that refers back (cue word) or is a bare noun
    phrase with no request verb/addressee is treated as a continuation
    fragment."""
    raw = _TOKEN_RE.findall(message.lower())
    content = [t for t in raw if len(t) >= 2 and t not in _STOPWORDS]
    has_marker = any(t in markers for t in raw)
    return has_cue(message, cues) or (len(content) <= 3 and not has_marker)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, minus stopwords and single characters."""
    return [
        t
        for t in _TOKEN_RE.findall(text.lower())
        if len(t) >= 2 and t not in _STOPWORDS
    ]


def bm25_scores(
    query_tokens: list[str],
    docs_tokens: list[list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """BM25 scores for query_tokens against each doc; IDF over the given docs."""
    n_docs = len(docs_tokens)
    if n_docs == 0:
        return []

    avgdl = sum(len(d) for d in docs_tokens) / n_docs or 1.0
    doc_freqs: Counter[str] = Counter()
    for doc in docs_tokens:
        for term in set(doc):
            doc_freqs[term] += 1

    query_terms = Counter(query_tokens)
    scores = []
    for doc in docs_tokens:
        doc_counts = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query_terms:
            tf = doc_counts.get(term, 0)
            if tf == 0:
                continue
            df = doc_freqs[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1)) / denom
        scores.append(score)
    return scores


def normalize(scores: list[float]) -> list[float]:
    """Divide by max; all zeros if max <= 0."""
    peak = max(scores) if scores else 0.0
    if peak <= 0:
        return [0.0] * len(scores)
    return [s / peak for s in scores]


def top_keywords(
    thread_tokens: list[str],
    other_threads_tokens: list[list[str]],
    k: int = 8,
) -> list[str]:
    """TF in this thread x 1/(1 + threads containing term); ties by (-score, token)."""
    tf = Counter(thread_tokens)
    scored = []
    for term, count in tf.items():
        containing = sum(1 for doc in other_threads_tokens if term in set(doc))
        scored.append((count / (1 + containing), term))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [term for _, term in scored[:k]]
