"""Build the router's bundled data files (dev-time only; not shipped).

    python tools/build_router_data.py            # uses tools/.cache when present
    python tools/build_router_data.py --refresh  # re-download everything

Writes src/branchy/router/data/:
  embeddings.i8, tokenizer.json   potion-base-8M (MIT), int8-quantised
  background.i32                  embeddings of 512 neutral chat prompts
  idf.json                        term rarity from the background corpus
  manifest.json, NOTICE           provenance and attribution

Background corpus = English, non-toxic user prompts from
  allenai/WildChat-1M (ODC-BY)      — real user->assistant chats
  OpenAssistant/oasst2 (Apache-2.0) — human-written prompts
Only derived numbers ship (vectors, term counts with df >= 2); prompt texts do
not. Needs network, the stdlib and `tokenizers`; no numpy, no LLM.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import struct
import sys
import time
import urllib.parse
import urllib.request
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from branchy.router.embed import StaticEmbedder  # noqa: E402
from branchy.router.text import term_set, topic_text  # noqa: E402

CACHE = ROOT / "tools" / ".cache"
OUT = ROOT / "src" / "branchy" / "router" / "data"

MODEL_REPO = "minishlab/potion-base-8M"
MODEL_FILES = ("model.safetensors", "tokenizer.json", "config.json")
OASST_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/main/"
    "2023-11-05_oasst2_prompts.messages.jsonl.gz"
)
WILDCHAT = "allenai/WildChat-1M"
WILDCHAT_ROWS = 837_989
WILDCHAT_PAGES = 90
PER_SOURCE = 3000
N_BACKGROUND = 512
MIN_CHARS, MAX_CHARS = 15, 400


def fetch(url: str, dest: Path, refresh: bool) -> Path:
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "branchy-build"})
    with urllib.request.urlopen(req, timeout=300) as r:
        dest.write_bytes(r.read())
    return dest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- model


def build_model(refresh: bool) -> dict:
    src = CACHE / "potion"
    for name in MODEL_FILES:
        fetch(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{name}", src / name, refresh)
    raw = (src / "model.safetensors").read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_len])
    info = header["embeddings"]
    assert info["dtype"] == "F32", info
    vocab, dim = info["shape"]
    start, end = info["data_offsets"]
    floats = array("f")
    floats.frombytes(raw[8 + header_len + start:8 + header_len + end])
    if sys.byteorder != "little":
        floats.byteswap()
    assert len(floats) == vocab * dim
    # one global scale keeps sums of rows exact integers in a common unit;
    # measured cosine error vs float32 is < 0.003
    scale = max(abs(x) for x in floats) / 127.0
    quant = array("b", (max(-127, min(127, round(x / scale))) for x in floats))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "embeddings.i8").write_bytes(quant.tobytes())
    (OUT / "tokenizer.json").write_bytes((src / "tokenizer.json").read_bytes())
    tok = json.loads((src / "tokenizer.json").read_text("utf-8"))
    skip = sorted(
        t["id"] for t in tok.get("added_tokens", []) if t.get("special")
    )
    return {
        "model": MODEL_REPO,
        "model_license": "MIT",
        "model_sha256": sha256(src / "model.safetensors"),
        "tokenizer_sha256": sha256(src / "tokenizer.json"),
        "vocab_size": vocab,
        "dim": dim,
        "quantization": "int8, one global max-abs scale",
        "scale": scale,
        "skip_token_ids": skip,
        "model_id": f"potion-base-8M-int8-{sha256(src / 'model.safetensors')[:8]}",
    }


# ---------------------------------------------------------------- corpus

_CODE_HINT = re.compile(r"```|[{};]\s*$|^\s*(def|class|import|SELECT|#include)\b", re.M)


def clean(text: str) -> str | None:
    text = " ".join(str(text).split())
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    if _CODE_HINT.search(text):
        return None
    ascii_share = sum(ch.isascii() for ch in text) / len(text)
    if ascii_share < 0.97:
        return None
    if len(term_set(text)) < 2:
        return None
    return text


def oasst_prompts(refresh: bool) -> list[str]:
    path = fetch(OASST_URL, CACHE / "oasst2_prompts.jsonl.gz", refresh)
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            if m.get("lang") != "en" or m.get("deleted") or m.get("role") != "prompter":
                continue
            if m.get("review_result") is False:
                continue
            t = clean(m.get("text", ""))
            if t:
                out.append(t)
    return out


def _rows(dataset: str, offset: int, length: int = 100) -> list[dict]:
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
        dict(dataset=dataset, config="default", split="train",
             offset=offset, length=length))
    req = urllib.request.Request(url, headers={"User-Agent": "branchy-build"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except Exception as exc:  # transient 5xx / rate limits
            wait = min(120, 5 * 2 ** attempt)
            print(f"  rows offset={offset}: {exc}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {dataset} offset {offset} (re-run to resume)")


def wildchat_prompts(refresh: bool) -> list[str]:
    """Pages are cached one by one, so an interrupted run resumes."""
    cache = CACHE / "wildchat_pages.json"
    pages: dict[str, list[str]] = (
        json.loads(cache.read_text("utf-8")) if cache.exists() and not refresh else {}
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    step = WILDCHAT_ROWS // WILDCHAT_PAGES
    for page in range(WILDCHAT_PAGES):
        if str(page) in pages:
            continue
        found = []
        for r in _rows(WILDCHAT, page * step):
            if r.get("language") != "English" or r.get("toxic"):
                continue
            users = [m["content"] for m in r["conversation"] if m["role"] == "user"]
            for u in users[:2]:
                t = clean(u)
                if t:
                    found.append(t)
        pages[str(page)] = found
        cache.write_text(json.dumps(pages), "utf-8")
        print(f"  wildchat page {page + 1}/{WILDCHAT_PAGES}: +{len(found)}")
        time.sleep(2)
    return [t for p in range(WILDCHAT_PAGES) for t in pages[str(p)]]


def select(texts: list[str], n: int) -> list[str]:
    """De-duplicate (case-insensitive) and take n by sha256 order — a
    deterministic, content-independent sample."""
    seen: dict[str, str] = {}
    for t in texts:
        seen.setdefault(t.lower(), t)
    ordered = sorted(seen.values(), key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    return ordered[:n]


NOTICE = """branchy router data — third-party material

embeddings.i8, tokenizer.json
  Derived from minishlab/potion-base-8M (https://huggingface.co/minishlab/potion-base-8M),
  a Model2Vec static embedding model by MinishLab, licensed under the MIT License.
  The float32 weights were quantised to int8. The MIT License requires this notice:

  Permission is hereby granted, free of charge, to any person obtaining a copy of
  this software and associated documentation files (the "Software"), to deal in
  the Software without restriction, including without limitation the rights to
  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
  the Software, and to permit persons to whom the Software is furnished to do so,
  subject to the following conditions: The above copyright notice and this
  permission notice shall be included in all copies or substantial portions of
  the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

background.i32, idf.json
  Derived statistics (embedding vectors and per-term document frequencies; no
  prompt text) computed from:
  - allenai/WildChat-1M (https://huggingface.co/datasets/allenai/WildChat-1M),
    Allen Institute for AI, licensed under the Open Data Commons Attribution
    License (ODC-BY) v1.0.
  - OpenAssistant/oasst2 (https://huggingface.co/datasets/OpenAssistant/oasst2),
    OpenAssistant / LAION, licensed under the Apache License 2.0.
"""


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    meta = build_model(args.refresh)
    # manifest first: StaticEmbedder reads it
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=2), "utf-8")

    oa = select(oasst_prompts(args.refresh), PER_SOURCE)
    wc = select(wildchat_prompts(args.refresh), PER_SOURCE)
    corpus = select(oa + wc, len(oa) + len(wc))
    print(f"corpus: {len(oa)} oasst2 + {len(wc)} wildchat = {len(corpus)}")

    # idf over the whole corpus; ship only terms seen in >= 2 prompts
    df: dict[str, int] = {}
    for t in corpus:
        for term in term_set(t):
            df[term] = df.get(term, 0) + 1
    n = len(corpus)

    def idf_milli(d: int) -> int:
        return round(1000 * math.log(1 + (n - d + 0.5) / (d + 0.5)))

    idf = {
        "n_docs": n,
        "formula": "ln(1 + (N - df + 0.5) / (df + 0.5)), x1000, rounded",
        "unknown_milli": idf_milli(0),
        "idf_milli": {t: idf_milli(d) for t, d in sorted(df.items()) if d >= 2},
    }
    (OUT / "idf.json").write_text(json.dumps(idf, separators=(",", ":")), "utf-8")

    # background vectors: N_BACKGROUND prompts, alternating sources
    emb = StaticEmbedder(OUT)
    picks = [x for pair in zip(oa, wc) for x in pair][:N_BACKGROUND]
    bg = array("i")
    for t in picks:
        bg.extend(emb.embed(topic_text(t)).values)  # same as the router
    if sys.byteorder != "little":
        bg.byteswap()
    (OUT / "background.i32").write_bytes(bg.tobytes())

    meta.update(
        corpus={
            "sources": {
                "allenai/WildChat-1M": {"license": "ODC-BY", "prompts": len(wc)},
                "OpenAssistant/oasst2": {"license": "Apache-2.0", "prompts": len(oa)},
            },
            "filters": f"English, non-toxic, {MIN_CHARS}-{MAX_CHARS} chars, no code, "
                       "case-insensitive dedup, sha256-ordered sample",
            "n_docs": n,
            "idf_terms": len(idf["idf_milli"]),
            "background_vectors": len(picks),
        }
    )
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=2), "utf-8")
    (OUT / "NOTICE").write_text(NOTICE, "utf-8")
    sizes = {p.name: p.stat().st_size for p in sorted(OUT.iterdir())}
    print(json.dumps(sizes, indent=2))


if __name__ == "__main__":
    main()
