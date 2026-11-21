"""Static sentence embeddings and the bundled routing data, in pure Python.

The embedding model is potion-base-8M (Model2Vec, MIT): one 256-dim vector
per tokenizer token; a text's embedding is the sum of its token vectors.
Vectors are stored as int8 with one global scale, so a text embedding is an
exact integer vector and a dot product is an exact integer. The only float
operations in a cosine are one sqrt and one division, which IEEE-754 rounds
identically on every platform — so similarities are bit-identical everywhere.

Bundled files (src/branchy/router/data/, built by tools/build_router_data.py):
  embeddings.i8   int8 matrix [vocab x dim], row-major
  tokenizer.json  the model's tokenizer (loaded with `tokenizers`)
  background.i32  int32 matrix [n x dim]: embeddings of neutral chat prompts
  idf.json        {"n_docs", "unknown_milli", "idf_milli": {term: int}}
  manifest.json   dims, hashes, provenance
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from array import array
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).with_name("data")
MAX_TOKENS = 512  # same cap Model2Vec applies by default


@dataclass(frozen=True)
class Vec:
    """An exact integer embedding and its squared norm."""

    values: tuple[int, ...]
    sq: int

    @classmethod
    def of(cls, values) -> "Vec":
        values = tuple(values)
        return cls(values, sum(v * v for v in values))


def dot(a: Vec, b: Vec) -> int:
    return sum(map(operator.mul, a.values, b.values))


def cosine(a: Vec, b: Vec) -> float:
    """Exact integer dot product; one correctly rounded sqrt and division."""
    if a.sq == 0 or b.sq == 0:
        return 0.0
    return dot(a, b) / math.sqrt(a.sq * b.sq)


class StaticEmbedder:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        from tokenizers import Tokenizer

        manifest = json.loads((data_dir / "manifest.json").read_text("utf-8"))
        self.dim: int = manifest["dim"]
        self.vocab_size: int = manifest["vocab_size"]
        raw = (data_dir / "embeddings.i8").read_bytes()
        if len(raw) != self.dim * self.vocab_size:
            raise ValueError("embeddings.i8 does not match manifest dims")
        self._matrix = array("b")
        self._matrix.frombytes(raw)
        self._tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
        self._skip_ids = frozenset(manifest.get("skip_token_ids", []))
        self.model_id: str = manifest["model_id"]

    def token_ids(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        return [i for i in ids if i not in self._skip_ids][:MAX_TOKENS]

    def embed(self, text: str) -> Vec:
        dim = self.dim
        m = self._matrix
        acc = [0] * dim
        for i in self.token_ids(text):
            start = i * dim
            acc = list(map(operator.add, acc, m[start:start + dim]))
        return Vec.of(acc)


class RouterData:
    """Everything the router reads from disk, loaded once per process."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.manifest = json.loads((data_dir / "manifest.json").read_text("utf-8"))
        self.embedder = StaticEmbedder(data_dir)

        idf = json.loads((data_dir / "idf.json").read_text("utf-8"))
        self.idf_milli: dict[str, int] = idf["idf_milli"]
        self.unknown_idf_milli: int = idf["unknown_milli"]

        dim = self.embedder.dim
        bg = array("i")
        bg.frombytes((data_dir / "background.i32").read_bytes())
        if bg.itemsize != 4 or len(bg) % dim:
            raise ValueError("background.i32 is malformed")
        self.background: list[Vec] = [
            Vec.of(bg[i:i + dim]) for i in range(0, len(bg), dim)
        ]
        digest = hashlib.sha256()
        for name in ("embeddings.i8", "tokenizer.json", "background.i32", "idf.json"):
            digest.update((data_dir / name).read_bytes())
        self.data_id: str = digest.hexdigest()[:12]

    def idf(self, term: str) -> int:
        return self.idf_milli.get(term, self.unknown_idf_milli)


@lru_cache(maxsize=4)
def load_router_data(data_dir: Optional[str] = None) -> RouterData:
    return RouterData(Path(data_dir) if data_dir else DATA_DIR)
