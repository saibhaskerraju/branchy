# ADR 0001: Deterministic lexical thread router ("sorting hat")

- Status: Accepted
- Scope: `src/branchy/router/` (`router.py`, `lexical.py`, `sticker.py`)

## Context

`BranchingChat` stores a conversation as a forest of user/assistant nodes. Every
incoming user message must be placed before the LLM is called: either as a
continuation of an existing branch (so the LLM receives only that branch's
history) or as the root of a new thread. The component that makes this
placement is the thread router.

The placement decision controls which history the LLM sees, so it has to be
cheap, explainable, reproducible across processes and runs, and safe to get
wrong.

## Decision

The router is pure, deterministic Python with no model calls, no network, and
no third-party dependency. It ranks candidate attachment points with BM25 plus a
recency term, then applies a small set of explicit rules to choose between
`NEW`, `ATTACH_FIRM` and `ATTACH_TENTATIVE`.

### Inputs

- The current tree (nodes, stickers, `seq`, `active_root_id`).
- The raw user message.
- `RouterConfig` (all thresholds below are its defaults).

### Candidates

Candidates are assistant nodes only, because a new user message is always
appended after an assistant reply:

- every **leaf** assistant (continue a branch), and
- every **root** assistant (the first reply of each thread; attaching here forks
  the thread at its origin).

A node that is both root and leaf counts once, as `leaf`. Candidates are
processed in `created_seq` order so that the output is independent of dict
iteration order.

### Candidate document

Each candidate is scored against one text:

```
<sticker of its thread> + <user text of its last turn> + <assistant text of its last turn>
```

The **sticker** is an extractive, LLM-free one-line summary of the whole thread,
rebuilt after every turn:

```
<first sentence of the root user message (<=160 chars)>
 | <up to 8 distinguishing keywords>
 | <first line of the most recent user message in the thread (<=160 chars)>
```

Keywords are ranked by `tf_in_thread / (1 + number_of_other_threads_containing_term)`,
ties broken alphabetically, so terms shared by every thread sink and terms unique
to this thread rise. The sticker gives a long thread a stable identity that a
single last turn would not.

### Scoring

1. **Tokenize** message and documents: lowercase `[a-z0-9_]+` tokens, drop
   single characters and a fixed English stopword list.
2. **BM25** (`k1 = 1.5`, `b = 0.75`) of the message against each candidate
   document, with IDF computed over the candidate set only. Scores are divided
   by the maximum so the best candidate has `bm25 = 1.0`; all-zero stays zero.
3. **Recency**: `0.5 ** (age / 3)`, where `age = tree.seq - last_used_seq` of
   the candidate's thread (`last_used_seq` is the sequence number of the last
   turn appended to that thread).
4. **Combined score** `s1 = bm25 + 0.3 * recency`.
5. **Evidence** = number of distinct message tokens that appear in the
   candidate document.

### Ranking and shortlist

Candidates are sorted by `(-s1, -thread last_used_seq, leaf before root,
-created_seq)`. Every tie-break is explicit so equal scores never depend on
insertion order.

**Dominated-root pruning.** A root candidate is removed from consideration when
its matched token set is a subset of the matched token set of any leaf in the
same thread. Forking at the root is only justified if the message hits content
that exists at the root and nowhere later; otherwise the thread is continued at
its leaf. Because the empty set is a subset of every set, a message with no
lexical overlap can never fork a root.

The top `k = 3` remaining candidates form the shortlist. `margin` is
`s1[best] - s1[second]` (or `None` if only one candidate exists).

### Decision rules

Rules are evaluated in this order.

1. **Empty tree** -> `NEW`.
2. **Low similarity** (`bm25[best] < 0.30`):
   - If the message is a **fragment**, never start a thread: attach
     `ATTACH_TENTATIVE` to the leaf of the active thread (the thread that
     received the previous turn); if that leaf is not an eligible candidate,
     attach to the best-ranked candidate.
   - Otherwise -> `NEW`.
3. **Firm vs tentative**: `ATTACH_FIRM` if `margin is None` or
   `margin >= 0.15`, else `ATTACH_TENTATIVE`.
4. **Single-term downgrade**: a would-be firm attachment with `evidence <= 1`
   and no referential cue word in the message is downgraded to
   `ATTACH_TENTATIVE`. One shared word (e.g. "recipe") is not enough to commit;
   a cue word ("it", "that", ...) signals that the user is referring back and
   keeps the firm decision.

**Fragment** is a lexical heuristic: the message contains a referential cue
word (`it`, `that`, `this`, `one`, `same`, `again`, `instead`, ...), or it has
at most three content tokens and no request marker (`tell`, `give`, `explain`,
`what`, `you`, `me`, ...). Both word lists are configurable via `RouterConfig`.

### Output

A `RoutingDecision` carrying the action, the target node, every candidate with
its `bm25`, `recency`, `s1` and `evidence`, the shortlist, the dominated roots,
the margin, and a human-readable `reason`. The decision is stored on the user
node it produced and is what the viewer shows under "Routing".

## Rationale

- **Determinism.** Same tree + same message -> same decision, in every process,
  on every run. Multiple workers sharing a store converge on identical trees,
  tests are exact, and every decision can be replayed and explained from the
  stored candidate table.
- **No LLM in the loop for placement.** Placement happens on every message and
  must not add latency, cost, non-determinism, or a second failure mode in
  front of the actual LLM call. The LLM is used only to produce the reply.
- **BM25 over embeddings.** Threads in one conversation are distinguished by
  specific nouns (table names, ingredients, error strings), which exact-match
  retrieval captures directly. The router stays pure stdlib and the wheel stays
  tiny.
- **Stickers give threads an identity.** Scoring only the last turn fails as
  soon as the user follows up on something said earlier in the thread; scoring
  the whole thread drowns the signal. The sticker is a bounded, deterministic
  middle ground that favours the thread's distinguishing vocabulary.
- **Leaf and root candidates, with dominated-root pruning.** Offering roots
  allows "go back to the beginning of that topic" forks, while the subset rule
  prevents a thread from fragmenting into forks every time its root words recur.
- **Fragments never open threads.** "How much salt?" has too little content to
  be a new topic; the least harmful placement is the thread the user was just
  in, marked tentative.
- **Tentative is a first-class outcome.** The router is a heuristic and will be
  wrong on genuinely ambiguous input. Marking low-margin and thin-evidence
  attachments as tentative keeps the error visible (viewer, `tentative` flag on
  the node) and leaves room for later re-parenting without changing the
  router.

## Consequences

- Routing cost is linear in the number of leaves and roots, with no I/O.
- Routing quality depends on the lexical content of the thread, including the
  assistant replies; with empty or constant assistant text the stickers carry
  little signal and more decisions land as tentative.
- Short referential messages following a topic switch can attach to the wrong
  thread; by construction they are tentative, never firm, and never create a
  spurious root.
- All thresholds (`k`, `w_rec`, `recency_half_life`, `new_sim_max`,
  `firm_margin`) and both word lists are exposed on `RouterConfig`; changing
  them changes decisions for every message routed afterwards but never
  retroactively.
- The router is English-centric: the stopword, cue and request-marker lists are
  English, and the tokenizer is ASCII-only.
