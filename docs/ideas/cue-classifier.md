# Idea: a trained cue classifier instead of hand-written cue lists

- Status: Idea, not scheduled (recorded 2026-11-21)
- Affects: `src/branchy/router/text.py` (cue lists), `RouterConfig` cue fields,
  [ADR 0002](../../decisions/0002-routing-algorithm.md) step 1
- If adopted, this becomes an ADR.

## Problem

The router's cues (`RETURN`, `NEW_TOPIC`, `ANAPHORA`, `FOLLOW_UP`, `FRAGMENT`)
come from about 340 hand-written English words and phrases, matched exactly on
whole words. They decide the highest-impact moves: whether a message may start
a new topic and whether it may fork. The lists are:

- **closed**: wording not on the list gets no cue ("circling back to",
  "about that Docker thing again", "different thing:");
- **typo-intolerant**: "goign back to" or "unrelatd" do not fire;
- **unmeasured beyond the eval set**: chosen by hand and checked only against
  the 67 turns in `eval/scenarios/`.

Bare affirmations ("yes", "sure", "go ahead") have no cue of their own. They
count as `FRAGMENT` and attach to the active thread, which is right only when the
assistant's question is in the latest reply.

## Why not an existing library

We searched Hugging Face, GitHub, PyPI and the research literature (2026-11-21)
for a ready-made, battle-tested component:

| Candidate | Why it does not fit |
|---|---|
| Topic-drift and dialogue topic-segmentation models | single-author research models; they segment a finished conversation into linear parts and do not say which earlier topic a message returns to |
| Dialogue-act classifiers (SwDA labels) | no standard maintained checkpoint; trained on phone speech; can tag "yes" but not what it answers |
| Discourse-marker lexicons | academic lists for written text; no chat phrases such as "going back to" |
| Coreference (fastcoref) | needs a transformer, trained on news-style text, unmaintained since 2023; resolves "it" inside one text, not across turns |

The industry answer is to ask an LLM (for example, rewriting the message into a
standalone question). branchy already does this for TENTATIVE placements
([ADR 0004](../../decisions/0004-llm-review-of-tentative-decisions.md)). For
deterministic, offline cues there is no off-the-shelf part.

## Proposal

Train a small classifier on the embeddings branchy already ships
(potion-base-8M, [ADR 0003](../../decisions/0003-static-embeddings-and-background-corpus.md)).
Model2Vec, the library behind that model, can train classifiers on top of its
static embeddings.

- **Input:** the user message (full text, cue words included).
- **Output labels**, per message, not exclusive:
  `RETURN`, `NEW_TOPIC`, `REFERS_BACK` (anaphora, follow-up, fragment) and
  `AFFIRMATION` (yes, sure, go ahead, do it).
- **Use in the router:** it replaces or supplements the cue flags in step 1.
  Thresholds and gates are unchanged. A low classifier confidence can mark the
  decision TENTATIVE so the LLM reviews it.
- **New behaviour from `AFFIRMATION`:** attach to the latest reply that ends
  with a question, or offer that reply to the LLM review as an option.

## What it would fix

- Phrasings not on the lists, and typos in cue phrases, if the training data
  includes them.
- Replies to an assistant question that was asked before a detour.

## Constraints to keep

- **Determinism.** A classifier head on integer embeddings must stay
  reproducible across machines (ADR 0003). Prefer integer or fixed-precision
  weights, or round the scores before any comparison, as the router already does.
- **Size and dependencies.** No new runtime dependency or download. The trained
  head ships in `src/branchy/router/data/` with its hash in `manifest.json`.
- **Safe failure.** Cue errors must keep failing towards "continue":
  a missed `RETURN` means no fork, and a missed `NEW_TOPIC` means attach.

## What is missing first

**A labelled dataset of chat messages with cue labels.** None is published;
this is the real work, not the model. Sources:

- real user turns from WildChat or oasst2 (already the background corpus
  sources; licence ODC-BY / Apache-2.0), labelled by hand;
- varied and misspelled versions of each phrase;
- negative examples: "that" used as grammar, "back" in a non-return sense
  ("my back hurts"), "earlier" as a topic word.

## How to judge it

Run `eval/run.py` with the classifier against the current lists. It must not
lower FORK precision or raise context loss. It should raise NEW recall and
exact-parent accuracy on new scenarios with varied wording and typos, written
**before** training and kept out of training data.

## Origin

This idea came from a review of the router on 2026-11-21. A 6-turn test showed
that a bare "yes" is placed correctly only by default, and that cue matching is
exact and English-only.
