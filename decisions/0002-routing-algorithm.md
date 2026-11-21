# ADR 0002: The routing algorithm

- Status: Accepted
- Scope: `src/branchy/router/router.py`, `src/branchy/router/text.py`
- Related: [ADR 0001](0001-message-tree-and-placement.md) (what is decided),
  [ADR 0003](0003-static-embeddings-and-background-corpus.md) (the data it reads)

## Context

[ADR 0001](0001-message-tree-and-placement.md) reduces routing to one choice:
the parent of the new user message — an assistant message, or none. This ADR
specifies how that choice is made. Every number below is a `RouterConfig`
field; the values are the defaults, chosen on the labelled set in `eval/`
(see "Evidence").

## Decision

### Definitions

| Term | Meaning |
|---|---|
| candidate | every assistant message `A`; its turn is `(U, A)` with `U` its user message |
| thread | all messages under one root user message |
| `path(A)` | the assistant messages from the thread root down to `A` — the history the LLM would see |
| `active_leaf` | the reply to the previous user message |
| `thread_leaf[T]` | the reply to the latest user message in thread `T` |
| `activity[T]` | before each message every thread is multiplied by `0.8409` (half-life 4 messages); the thread that receives the message gets +1 |

All three are derived from the stored messages in creation order (ADR 0001).

### Step 1 — Read the message

- **Topic words:** lowercase words, minus stop words, minus conversation
  meta-phrases (RETURN and NEW_TOPIC phrases such as "going back to",
  "unrelated question" — they describe the conversation, not its topic).
  **Terms** are the Porter stems of the topic words.
- **Embedding** of the topic words only. Function words carry the style of a
  request ("how do I … should I"), not its topic; embedding them made unrelated
  how-to questions look alike.
- **Cue flags** (closed English lists in `text.py`):

| Cue | Fires when |
|---|---|
| `RETURN` | a phrase like *go back to, back to the, earlier, you mentioned, originally* |
| `NEW_TOPIC` | a phrase like *unrelated, btw, different question, switching gears* |
| `ANAPHORA` | *it, that, this, them, those, ones, same, another, he, she, his, her, their, there, here, former, latter, above, mentioned…* or "a/the/that … one", in a message of ≤ 6 topic words (in longer messages "that" is usually a relative pronoun) |
| `FOLLOW_UP` | *an example, elaborate, more detail, in Python, simpler…* or an opening *And / Also / But / So / What about / How about*, in a message of ≤ 6 topic words |
| `FRAGMENT` | ≤ 3 topic words and no request word (*how, why, what, can, should, explain, want, need, understand…*) |
| `AFFIRMATION` | opens with a yes/no-style answer (*yes, sure, ok, no, go ahead, sounds good…*), in a message of ≤ 6 topic words |

`ANAPHORA`, `FOLLOW_UP`, `FRAGMENT` and `AFFIRMATION` together mean **refers back**.

### Step 2 — Score every candidate

```
lex(A) = Σ idf(t) for t in terms(m) ∩ terms(U ∪ A)  /  Σ idf(t) for t in terms(m)
rest   = the message's topic words whose stem is NOT in that intersection
sem(A) = clip( (max(cos(rest, U), cos(rest, A)) − bg(rest)) / 0.25 , 0, 1 )
rel(A) = lex + (1 − lex) · sem
```

- `idf` comes from the background corpus (fixed, integer); `bg(x)` is the 95th
  percentile cosine of `x` against 512 background prompts — "how similar is this
  to text it has nothing to do with".
- **The two channels are complementary, not added.** A static embedding is
  close to a weighted bag of words, so a single shared salient word
  ("container", "index") would otherwise be counted by `lex` *and* dominate
  `sem`. `lex` is the share of the message explained by exact shared words;
  `sem` judges only the rest. `rel` reads as "how much of the message this turn
  accounts for": matched words fully, the rest as far as it is related in
  meaning. No mixing weight.
- All scores are absolute (never normalised against other candidates) and
  rounded to 6 decimals before any comparison.

### Step 3 — Explicit overrides

`reply_to=A` → parent `A`; `new_topic=True` → NEW. Both `FIRM`,
`decided_by="explicit"`. An empty tree → NEW.

### Step 4 — New topic or attach (the NEW gate)

`best = max rel(A)`. Rules in order:

| # | Condition | Result |
|---|---|---|
| a | `NEW_TOPIC` cue and best < 0.60 | **NEW** |
| b | refers back and no `RETURN` cue, **or** a `RETURN` cue with best < 0.35 | never NEW (a return points at something already said, even when its words do not match it). Thread = the active thread, unless another thread's best rel leads it by ≥ 0.15 **and** is itself ≥ 0.35 (ADR 0004). TENTATIVE when the chosen thread's best rel < 0.35, **except** a `FRAGMENT` or `AFFIRMATION` in the active thread right after an assistant reply that asks a question (a `?` in its last 300 characters): that answers the question and is FIRM (gate `answers_question`) |
| c | best ≥ 0.35 | attach; thread = best `score` among threads with best rel ≥ 0.35 |
| d | best < 0.25 | **NEW** |
| e | 0.25 ≤ best < 0.35 | uncertain: attach to the active thread if the best node is in it; else to the best node's thread if they share a rare term (idf ≥ 7000, ADR 0004); else NEW. Always TENTATIVE |

Rule (e) prefers attaching because a wrong attach only adds context while a wrong
NEW removes all of it; on the eval set it is the only band rule with zero
context loss (see "Evidence").

### Step 5 — Which thread

```
score(T) = max rel in T  +  0.15 · activity[T] / max activity
```

Activity only ranks threads that already passed the gate; it never opens it.

### Step 6 — Where in the thread

```
base = thread_leaf[T]             X = argmax rel in T   (ties → newer)
```

| Where X is | Check | Result |
|---|---|---|
| `X = base` | — | continue at base |
| X is an ancestor of base | fork gate (X, base) | FORK at X, else continue at base |
| X on another branch | X's lead over the **unshared** part of the current branch ≥ 0.15, or a `RETURN` cue | `leaf_X` = newest leaf under X: FORK at X if fork gate (X, leaf_X) passes, else resume at `leaf_X` |
|  | otherwise | continue at base |

"Unshared": the common ancestors of the two branches are in the history either
way, so they are left out of the comparison.

The action is read off the chosen parent: the active leaf → `CONTINUE`;
another leaf → `SWITCH`; a node with children → `FORK`; none → `NEW`.

### Step 7 — The fork gate

With `k` = the number of turns below X that the fork would drop:

```
δ(k) = min(0.30, 0.10 + 0.03·(k − 1))
G1  distinct evidence:  rel(X) − max rel of the dropped turns  ≥ δ(k)
G2  intent:             a RETURN cue
G3  not a FRAGMENT
FORK only if G1 ∧ G2 ∧ G3
```

A fork hides the dropped turns from the LLM, so it needs both evidence (G1) and
the user saying they are going back (G2). An earlier draft also allowed a fork
without a cue when G1 held at 2·δ and rel(X) ≥ 0.6; on the eval set both forks
it produced were wrong (the user mentioned something from an earlier answer
without abandoning the later turns), so it was removed. "Instead" is not a fork
cue either: in "use map instead of the for loop" the old answer's code is still
needed.

### Step 8 — Confidence

Every deciding check is a comparison `x ≥ θ` with slack `|x − θ|`. The decision
is `FIRM` when every deciding check has slack ≥ 0.05 and neither the uncertain
band (4e) nor the low-evidence reference fallback (4b) was used; otherwise
`TENTATIVE`, and the reason names the closest check. All checks, with values,
thresholds and slacks, are recorded in `RoutingDecision.gates`.

### Determinism

Integer embeddings and dot products, integer IDF, a constant decay factor, scores
rounded to 6 decimals, ties broken by creation order (ADR 0003). The same tree
and message give the same decision on any machine; tests check it with a fresh
router instance.

## Evidence

`eval/scenarios/handwritten.json`: 11 conversations, 46 turns, labels written
for this design (to be reviewed). Teacher-forced (`python eval/run.py`):

| Metric | Value |
|---|---|
| exact parent | 93.5% (43/46) |
| context loss (gold parent not in the history sent) | 3.6% (1/28) |
| NEW precision / recall | 100% / 94.4% |
| FORK precision | 100% (2/2) |
| errors that were FIRM | 0 of 3 |
| TENTATIVE share | 30.4% |

How the main choices were made on that set:

| Choice | Alternative | Result |
|---|---|---|
| embed topic words only | embed full text | measured at an earlier stage of the design: context loss 18.5% → 14.8%, NEW precision 87.5% → 93.3% |
| `rel = lex + (1−lex)·sem`, sem on the rest | `0.6·sem + 0.4·lex`, sem on full text | coverage has a wide flat optimum (tau_new 0.25; s_full 0.20–0.30; tau_attach 0.30–0.45 all give the best result); the additive mix reaches it only at the edge of the grid |
| band → active thread | band → NEW / band → attach only with a shared word | 0 context loss vs 1–4 / 1–2 |
| fork needs RETURN cue | strong evidence may fork without a cue | 2 wrong forks removed |

The set is small and was used to choose the defaults, so these numbers are
optimistic; they are a regression baseline, not a quality claim. Growing it
(real WildChat conversations, labelled by hand) is the next step.

## Known limits

- A static embedding has no world knowledge beyond word co-occurrence: "a soft
  one" after an interruption is not linked to "boil an egg"; "pot size for
  tomatoes" can get a small lift from "slim image". These land TENTATIVE.
- Paraphrase with no shared words ("ship it" / "deploy") gets little semantic
  credit.
- Cue lists are English only and hand-written.
