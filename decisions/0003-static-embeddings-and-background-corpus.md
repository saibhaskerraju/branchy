# ADR 0003: Static embeddings and the background corpus

- Status: Accepted
- Scope: `src/branchy/router/embed.py`, `src/branchy/router/data/`,
  `tools/build_router_data.py`
- Related: [ADR 0002](0002-routing-algorithm.md)

## Context

Word overlap alone (the 0.3.x router) fails in both directions: it misses
continuations that share meaning but not words ("deploy" / "ship to production"),
and it joins unrelated topics that share a word ("index" in Postgres and in index
funds). Routing also needs an *absolute* notion of "unrelated", which a
max-normalised score cannot give.

Constraints: one package, no LLM, no network at runtime, deterministic across
machines, and few dependencies.

## Decision

### Model: potion-base-8M, int8, pure Python

- **Model:** `minishlab/potion-base-8M` (Model2Vec, MIT). A static model: one
  256-dim vector per token of the BGE-base-en-v1.5 tokenizer (29,528 tokens), so
  embedding a text is "tokenise, add up the token vectors". No transformer runs.
- **Quantised to int8 with one global scale** (max |w| / 127). Measured against the
  float32 model, cosines move by less than 0.003.
- **Pure Python, no numpy.** A text embedding is a sum of int8 rows — an exact
  integer vector; a dot product is an exact integer. A cosine then needs one
  `sqrt` and one division, which IEEE-754 rounds identically on every platform.
  Similarities are therefore bit-identical on any CPU and OS, which numpy (SIMD
  reductions in CPU-dependent order) would not guarantee. Measured cost: ~4 ms to
  embed a 130-token text, ~7.5 µs per 256-dim dot product.
- The router embeds a message's **topic words** (stop words and conversation
  meta-phrases removed; [ADR 0002](0002-routing-algorithm.md) step 1), and the
  background prompts are embedded the same way. Texts are capped at 512 tokens,
  as Model2Vec does by default. Each message is embedded once and cached by
  message id (contents never change). Measured: ~30 ms per route on a 500-node
  tree; ~200 ms once per process to embed an existing tree.
- **Dependency:** `tokenizers` to run the model's tokenizer. litellm already
  depends on it, so it adds nothing to an install; it is declared explicitly.

### Background corpus

A fixed set of real, neutral chat prompts provides two absolute references:

1. **Term rarity (IDF)** for the lexical score. Computed only from the corpus and
   shipped as integers, so a word's weight never drifts as a tree grows. Common
   chat words ("good", "index", "use") weigh little; specific ones ("postgres",
   "sourdough", `pg_stat_statements`) weigh a lot. Unknown words get the maximum.
2. **"Unrelated" similarity**: the 95th-percentile cosine between a message and
   512 background prompts. The semantic score is the lift *above* that, which
   works for any message without a fixed cosine threshold.

**Sources** (public, permissive, real users):

| Dataset | Licence | Used |
|---|---|---|
| `allenai/WildChat-1M` | ODC-BY | first two user turns of English, non-toxic conversations sampled across the dataset |
| `OpenAssistant/oasst2` | Apache-2.0 | English human-written prompts |

Filters: 15–400 characters, no code, ≥ 2 content terms, case-insensitive
de-duplication, a deterministic sha256-ordered sample of up to 3,000 per source.
Rejected candidates: lmsys-chat-1m (gated, custom licence), no_robots
(non-commercial), dolly-15k and natural_questions (share-alike), quora
duplicates (no licence stated), hh-rlhf (red-team content).

**Only derived numbers ship:** background vectors and per-term document
frequencies for terms seen in at least two prompts. Prompt texts are not in the
package. Attribution is in `src/branchy/router/data/NOTICE`; counts, hashes and
filters are in `manifest.json`.

### Reproducible build

`tools/build_router_data.py` (dev-time, not shipped) downloads the model and the
datasets, quantises, filters, and writes `src/branchy/router/data/`. It uses the
stdlib and `tokenizers` only, caches downloads in `tools/.cache/`, and resumes
after rate limits. Each decision records `data_id`, a hash of the data files.

## Consequences

- The wheel grows by about 9 MB (embeddings 7.6 MB, tokenizer 0.7 MB, IDF and
  background under 1 MB).
- Routing quality is bounded by a static model: good at topic, weaker at
  paraphrase and blind to word order and negation. The lexical score and the cue
  rules ([ADR 0002](0002-routing-algorithm.md)) cover what it misses.
- Changing the model or corpus changes decisions; `data_id` on every decision
  makes that visible, and `eval/run.py` measures it.
