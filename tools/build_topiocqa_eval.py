"""Build router eval scenarios from TopiOCQA (dev-time; not shipped).

TopiOCQA (McGill-NLP, TACL 2022) is information-seeking conversational QA with
topic switching: https://huggingface.co/datasets/McGill-NLP/TopiOCQA
Licence: CC BY-NC-SA 4.0. The generated files are derived data under that
licence, so they are written to eval/scenarios-topiocqa/ and git-ignored; this
script is how to recreate them.

Writes two scenario files in the eval/scenarios format (see eval/README.md):

  chains.json       one TopiOCQA conversation per scenario. Every turn after the
                    first continues the previous one (gold = previous turn).
                    TopiOCQA's topic switches are drill-downs ("which hormones
                    are related to it?", "the third one"), not new chats, so a
                    topic change is NOT labelled NEW. Tests that the router
                    does not start a new topic or jump away during drill-downs.

  interleaved.json  two unrelated TopiOCQA conversations interleaved in blocks
                    of BLOCK turns (A1-4, B1-4, A5-8, ...). The first turn of
                    each conversation is NEW; the first turn of each later block
                    returns to that conversation's last turn (a SWITCH); the
                    rest continue. Tests NEW and returning to an earlier topic.
                    The switches are splice points, so they are easier than
                    natural ones.

Assistant replies are TopiOCQA's answer plus its rationale (the supporting
passage sentence), which is closer to an LLM reply than the short answer alone.

Usage:  python tools/build_topiocqa_eval.py [--split valid|train] [--refresh]
Stdlib only. Downloads are cached in tools/.cache/topiocqa/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tools" / ".cache" / "topiocqa"
OUT = ROOT / "eval" / "scenarios-topiocqa"
BASE = "https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/main/data/"
SHA256 = {  # pinned so the eval is reproducible
    "valid": "1bba9512b24b2e5de22704766dd80b1dc497bb7262799f66c3912e6f413ac1c6",
    "train": "d976c118996b238d1c689342f7abe28b0f6033aa59787efe5f1c6c26dbb8c39c",
}
BLOCK = 4
UNANSWERABLE = "I couldn't find an answer to that."


def fetch(split: str, refresh: bool) -> Path:
    dest = CACHE / f"topiocqa_{split}.jsonl"
    if refresh or not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}topiocqa_{split}.jsonl"
        print(f"download {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "branchy-build"})
        with urllib.request.urlopen(req, timeout=300) as r:
            dest.write_bytes(r.read())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != SHA256[split]:
        raise SystemExit(f"{dest}: sha256 {digest} does not match the pinned {SHA256[split]}")
    return dest


def load(path: Path) -> dict[int, list[dict]]:
    convs: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            convs[row["Conversation_no"]].append(row)
    for turns in convs.values():
        turns.sort(key=lambda r: r["Turn_no"])
    return dict(sorted(convs.items()))


def reply(row: dict) -> str:
    answer = row["Answer"].strip()
    if answer.upper() == "UNANSWERABLE":
        return UNANSWERABLE
    rationale = row.get("Rationale", "").strip()
    if rationale and rationale.lower() != answer.lower():
        return f"{answer}\n\n{rationale}"
    return answer


# A question that points back past the previous turn ("the country mentioned
# previously", "the first one"). For these, an earlier turn whose text shares a
# word with the question is also accepted as the parent (a fork to it keeps
# what the user points at). Gold stays the previous turn.
POINTER = re.compile(
    r"\b(mentioned|previously|earlier|before|above|former|latter|aforesaid|"
    r"aforementioned|back to|(first|second|third|last) one)\b"
)
_WORD = re.compile(r"[a-z0-9]+")
_NOT_KEY = frozenset(
    "mentioned previously earlier before above former latter aforesaid "
    "aforementioned first second third which where there their about other "
    "could would should these those what when were does have".split()
)


def keywords(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) >= 5 and w not in _NOT_KEY}


def accept_for(row: dict, history: list[tuple[str, dict]]) -> list[str]:
    """Earlier turns (not the previous one) a pointer question may refer to."""
    question = row["Question"].lower()
    if not POINTER.search(question) or len(history) < 2:
        return []
    keys = keywords(question)
    return [tid for tid, past in history[:-1]
            if keys & keywords(past["Question"] + " " + reply(past))]


def turn(tid: str, gold: str, row: dict, history: list[tuple[str, dict]]) -> dict:
    out = {"id": tid, "gold": gold, "user": row["Question"].strip(), "assistant": reply(row)}
    accept = accept_for(row, history)
    if accept:
        out["accept"] = accept
    return out


def chains(convs: dict[int, list[dict]], split: str) -> list[dict]:
    out = []
    for no, rows in convs.items():
        turns, prev, history = [], "NEW", []
        for row in rows:
            tid = f"t{row['Turn_no']}"
            turns.append(turn(tid, prev, row, history))
            history.append((tid, row))
            prev = tid
        topics = list(dict.fromkeys(r["Topic"] for r in rows))
        out.append({
            "id": f"topiocqa-{split}-{no}",
            "note": f"TopiOCQA {split} conversation {no}; topics: {' > '.join(topics)}",
            "turns": turns,
        })
    return out


def interleaved(convs: dict[int, list[dict]], split: str) -> list[dict]:
    """Pair conversations in order, skipping a partner that shares a topic."""
    pool = list(convs.items())
    out = []
    while len(pool) >= 2:
        no_a, rows_a = pool.pop(0)
        topics_a = {r["Topic"] for r in rows_a}
        idx = next((i for i, (_, rows) in enumerate(pool)
                    if topics_a.isdisjoint(r["Topic"] for r in rows)), None)
        if idx is None:
            continue
        no_b, rows_b = pool.pop(idx)
        last = {"a": "NEW", "b": "NEW"}
        cursor = {"a": 0, "b": 0}
        rows = {"a": rows_a, "b": rows_b}
        history: dict[str, list[tuple[str, dict]]] = {"a": [], "b": []}
        turns = []
        while cursor["a"] < len(rows_a) or cursor["b"] < len(rows_b):
            for side in ("a", "b"):
                for row in rows[side][cursor[side]:cursor[side] + BLOCK]:
                    tid = f"{side}{row['Turn_no']}"
                    turns.append(turn(tid, last[side], row, history[side]))
                    history[side].append((tid, row))
                    last[side] = tid
                cursor[side] += BLOCK
        out.append({
            "id": f"topiocqa-{split}-{no_a}x{no_b}",
            "note": (f"TopiOCQA {split} conversations {no_a} (a) and {no_b} (b), "
                     f"interleaved in blocks of {BLOCK}"),
            "turns": turns,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SHA256), default="valid")
    ap.add_argument("--refresh", action="store_true", help="download again")
    args = ap.parse_args()
    convs = load(fetch(args.split, args.refresh))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("chains", chains(convs, args.split)),
                       ("interleaved", interleaved(convs, args.split))):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
        n_turns = sum(len(c["turns"]) for c in data)
        n_new = sum(t["gold"] == "NEW" for c in data for t in c["turns"])
        n_acc = sum("accept" in t for c in data for t in c["turns"])
        print(f"{path.relative_to(ROOT)}: {len(data)} conversations, {n_turns} turns, "
              f"{n_new} NEW, {n_acc} with accepted alternatives")


if __name__ == "__main__":
    main()
