"""Porter stemmer (M.F. Porter, 1980, "An algorithm for suffix stripping").

The original algorithm, implemented without dependencies so the router stays
pure Python and byte-for-byte reproducible. Input is expected lowercase.
"""

from __future__ import annotations

from functools import lru_cache

_VOWELS = frozenset("aeiou")


def _is_cons(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_cons(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """m in [C](VC)^m[V]."""
    m = 0
    i = 0
    n = len(stem)
    while i < n and _is_cons(stem, i):
        i += 1
    while i < n:
        while i < n and not _is_cons(stem, i):
            i += 1
        if i >= n:
            break
        while i < n and _is_cons(stem, i):
            i += 1
        m += 1
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_cons(stem, i) for i in range(len(stem)))


def _ends_double_cons(word: str) -> bool:
    return (
        len(word) >= 2
        and word[-1] == word[-2]
        and _is_cons(word, len(word) - 1)
    )


def _cvc(word: str) -> bool:
    """consonant-vowel-consonant ending, last consonant not w, x or y."""
    if len(word) < 3:
        return False
    if not (
        _is_cons(word, len(word) - 3)
        and not _is_cons(word, len(word) - 2)
        and _is_cons(word, len(word) - 1)
    ):
        return False
    return word[-1] not in "wxy"


def _replace(word: str, suffix: str, repl: str, min_m: int) -> tuple[str, bool]:
    """(new word, matched) — matched means the suffix was present, whether or
    not the measure condition allowed the replacement."""
    if not word.endswith(suffix):
        return word, False
    stem = word[: len(word) - len(suffix)]
    if _measure(stem) > min_m:
        return stem + repl, True
    return word, True


def _step1a(w: str) -> str:
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies"):
        return w[:-2]
    if w.endswith("ss"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def _step1b(w: str) -> str:
    if w.endswith("eed"):
        stem = w[:-3]
        return w[:-1] if _measure(stem) > 0 else w
    for suffix in ("ed", "ing"):
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if not _has_vowel(stem):
                return w
            w = stem
            if w.endswith(("at", "bl", "iz")):
                return w + "e"
            if _ends_double_cons(w) and w[-1] not in "lsz":
                return w[:-1]
            if _measure(w) == 1 and _cvc(w):
                return w + "e"
            return w
    return w


def _step1c(w: str) -> str:
    if w.endswith("y") and _has_vowel(w[:-1]):
        return w[:-1] + "i"
    return w


_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
)

_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
)

_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)


def _step_table(w: str, table) -> str:
    for suffix, repl in table:
        new, matched = _replace(w, suffix, repl, 0)
        if matched:
            return new
    return w


def _step4(w: str) -> str:
    for suffix in _STEP4:
        if not w.endswith(suffix):
            continue
        stem = w[: len(w) - len(suffix)]
        if suffix == "ion" and not (stem.endswith("s") or stem.endswith("t")):
            return w
        if _measure(stem) > 1:
            return stem
        return w
    return w


def _step5(w: str) -> str:
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            w = stem
    if _measure(w) > 1 and _ends_double_cons(w) and w.endswith("l"):
        w = w[:-1]
    return w


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    """Porter stem of a lowercase word. Words of 1-2 letters are returned as is."""
    if len(word) <= 2 or not word.isalpha():
        return word
    w = _step1a(word)
    w = _step1b(w)
    w = _step1c(w)
    w = _step_table(w, _STEP2)
    w = _step_table(w, _STEP3)
    w = _step4(w)
    w = _step5(w)
    return w
