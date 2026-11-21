# ADR 0004: LLM review of TENTATIVE routing decisions

- Status: Accepted
- Scope: `src/branchy/router/judge.py`, `src/branchy/llm.py`, `src/branchy/chat.py`,
  `eval/run.py --llm`
- Related: amends [ADR 0001](0001-message-tree-and-placement.md) ("No LLM in
  routing"); builds on [ADR 0002](0002-routing-algorithm.md)

## Context

ADR 0001 kept LLMs out of routing, but left a condition for revisiting that: "if
errors cluster in TENTATIVE decisions, a fallback can be reconsidered with data."

A live session with a real model (2026-11-14) showed two failure patterns the
handwritten set did not cover, both TENTATIVE:

- **A reference taken by a generic answer.** "can we use fastapi in that" right
  after a Docker answer was moved to an earlier "who are you" thread. The generic
  self-introduction was loosely similar to "fastapi" (rel 0.314), and the long
  Docker answer was not (0.000).
- **A rare name diluted.** "does fastapi offer rate limiting feature" shared only
  `fastapi` with its thread. Generic words around it shared the idf total
  (lex 0.268), so it fell into the uncertain band outside the active thread → NEW.

Those turns and five handwritten conversations with the same patterns on other
topics were added to `eval/scenarios/` (`live.json`,
`reference-and-rare-term.json`) **before** any change. All 67 turns: 58/67
exact, 9 errors, **all TENTATIVE**.

## Decision

### 1. Fix the rules first (deterministic, ADR 0002 updated)

| Change | Why |
|---|---|
| A reference may switch to another thread only if that thread also clears `tau_attach` (gate `thread_switch_evidence`) | a lead over a weak active thread is not evidence; generic answers lead often |
| In the uncertain band, a best match outside the active thread attaches there if it shares a **rare** term (idf ≥ `idf_rare` = 7000, i.e. in < ~1/1100 background prompts); otherwise NEW as before | names like `fastapi`, `redis`, `alembic` identify a topic; the false matches NEW guards against share common words (`container` 5250, `index` 6685, `python` 3720) |
| `want need understand learn know` are request markers | "i want to understand docker in brief" was a FRAGMENT and therefore could never be NEW |

Rejected: using the whole-message meaning as a floor on `rel` (fixes the rare-name
case but counts the shared word twice). It turned the tomato and index-fund traps
into **FIRM** errors, which an LLM step limited to TENTATIVE decisions could
never catch.

Each decision now records **why** it is TENTATIVE in
`RoutingDecision.uncertainty`: `band` (rel in [0.25, 0.35)), `reference`
(attached by a reference cue on weak evidence) or `margin` (a deciding check
passed by less than 0.05).

### 2. The chat's LLM reviews every TENTATIVE decision, and only those

`BranchingChat` requires a `model`. The same model writes the replies and
reviews routing: `RoutingJudge` runs after the router on **every TENTATIVE
decision and never on a FIRM one**. There is no switch. A first version was
opt-in (`routing_judge=`) and allowed a custom `responder=` callable instead of
a model. Both were removed (breaking change): with a custom responder there was
no model to review routing, and an off switch would have left the same TENTATIVE
errors in place. `add_message(..., assistant_message=...)` still lets an app
supply the reply itself; routing is reviewed the same way.

- **Closed choice.** The options are each thread's latest answer (up to 3, most
  recent first, so option A is always "the most recent exchange"), the router's
  own pick if it is something else (e.g. a fork), and N = new topic. The router's
  pick is not marked, so the model judges on its own. The model answers with one
  letter.
- **No new forks.** The LLM can only choose a fork the router proposed. Forks
  hide context the user cannot see (ADR 0001), so they stay with the rules.
- **Failure keeps the rules.** An error, timeout (10 s default) or answer
  without a valid letter keeps the router's decision and records the error.
- **Auditable.** `RoutingDecision.llm` stores the rules' action and target, the
  options, the raw answer, latency and model. `decided_by="llm"` only when the
  model changed the placement. The viewer shows `llm agreed` / `llm override` /
  `llm failed` badges and the review in the detail pane.
- **Confidence is unchanged.** A reviewed decision stays TENTATIVE: the rules
  were unsure, and the model is a second opinion, not proof.

## Evidence

All 67 turns (handwritten 46, live 7, reference-and-rare-term 14), teacher-forced:

| | before (ADR 0002 rules) | rules fixed | rules fixed + LLM |
|---|---|---|---|
| exact parent | 86.6% (58/67) | 94.0% (63/67) | **98.5% (66/67)** |
| context loss | 14.7% (5/34) | 2.9% (1/34) | **0.0% (0/34)** |
| NEW precision / recall | 90.9% / 90.9% | 100% / 93.9% | 100% / 100% |
| errors FIRM / TENTATIVE | 0 / 9 | 0 / 4 | 0 / 1 |
| TENTATIVE share | 32.8% | 31.3% | 31.3% |

The handwritten set alone is unchanged by the rule fixes (43/46).

LLM step: 21 decisions reviewed, 0 failures; it **fixed 3, broke 0**. The
remaining error (`resume-cover-letter/t3`) needs a fork the router did not
propose, which the judge cannot choose by design.

Latency, measured with the model in `LLM_MODEL` at the time: ~0.75–0.95 s per
review after the first call (~3.7 s with connection setup), added before the
reply on about a third of messages.

These numbers come from a small set, partly written while making the fixes, and
one model. They show the direction, not the quality you will get. The answers
are cached in `eval/llm_cache/` so the run is reproducible.

## Consequences

- FIRM decisions stay deterministic. TENTATIVE decisions depend on the model.
  Stored placements never change: the router rebuilds its state from the tree,
  not from past decisions. `ThreadRouter.route()` on its own is still pure;
  tests and `eval/run.py` (without `--llm`) use it directly.
- No offline mode: a chat always needs a model (the old `echo_responder` is
  gone). Tests replace `litellm.completion` with a fake.
- Cost: one short extra call per TENTATIVE message, with conversation excerpts
  in the prompt, sent to the same provider as the replies.
- Not done yet: running the judge **in parallel** with a reply generated on the
  router's choice, and regenerating only on disagreement. That would hide the
  latency when the model agrees, which was 18 of 21 times here.
