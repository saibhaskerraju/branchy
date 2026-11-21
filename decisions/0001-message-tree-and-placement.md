# ADR 0001: The message tree and what the router decides

- Status: Accepted
- Scope: `src/branchy/tree.py`, `src/branchy/models.py`, `src/branchy/chat.py`,
  `src/branchy/router/`
- Related: [ADR 0002](0002-routing-algorithm.md) (how the router decides),
  [ADR 0003](0003-static-embeddings-and-background-corpus.md) (what it reads)

## Context

A linear chat history is the wrong data structure for a real conversation.
People interleave topics, return to earlier answers and abandon directions.
With a flat transcript an application has two bad options: send the LLM
everything (cost, latency, distraction), or send a window (and silently drop
the message the user is referring to).

`branchy` stores the conversation as a tree and gives the LLM exactly one
path: the chain of messages from the new message's parent back to its root.
Choosing the context becomes choosing a place in the tree.

## Decision

### A node is a message

Every message — user or assistant — is one node. Each user message gets exactly
one assistant reply, which is its only child. So:

- **branching only ever happens at assistant messages**;
- **sisters are always user messages** (children of the same assistant message);
- a new topic is a new root user message.

The viewer and the terminal UI draw a synthetic "conversation" node above all
roots so the picture is a single tree; nothing extra is stored.

### The router chooses one parent

For each new user message the router returns one parent: an existing
assistant message, or none. Every outcome is that one choice:

| Parent chosen | Action | Shape |
|---|---|---|
| the active leaf (the reply to the previous message) | `CONTINUE` | child — the chain grows |
| the leaf of another branch or thread | `SWITCH` | child at the end of that branch — resume it |
| an assistant message that already has a reply | `FORK` | **sister** of the existing user children |
| none | `NEW` | new root |

The action is derived from the chosen parent, not decided separately.

### Fan-out is unbounded

An assistant message has as many children as the number of times the user went
back to it and took a different direction. There is no cap: a binary limit would
buy nothing (edges mean "continues from", not an ordering to search by) and would
force a wrong placement on the third return. Children are ordered by time.

### The cost asymmetry decides close calls

Placing a message deeper than needed only adds context the LLM can ignore.
Placing it higher (a fork) **removes** the messages below the fork point from the
LLM's history, and the user cannot see that they are gone. So continuing is the
default and a fork must earn it ([ADR 0002](0002-routing-algorithm.md), step 7).

### No LLM in routing

> Amended by [ADR 0004](0004-llm-review-of-tentative-decisions.md): the condition
> below was met (all errors TENTATIVE), so the chat's LLM now reviews every
> TENTATIVE decision. FIRM decisions and `ThreadRouter` itself are unchanged.

The router is deterministic: same tree + same message gives the same decision in
any process on any machine. Uncertain decisions are flagged `TENTATIVE` rather
than escalated to a model. Rationale:

- the default for uncertainty (continue) never loses context, so an ambiguous
  case is cheap;
- explicit placement (`reply_to`, `new_topic`) lets the user or the app settle it;
- an LLM fallback would add a network call before every reply, latency,
  non-reproducible decisions and a prompt to maintain, for a gain nobody has
  measured. `eval/run.py` reports how many errors fall in `TENTATIVE`
  decisions; if they cluster there, a fallback can be reconsidered with data.

### Explicit placement

`BranchingChat.add_message(text, reply_to=<assistant id>)` places the message
under that answer without routing; `new_topic=True` starts a new root. Both are
recorded with `decided_by="explicit"`.

### Router state is derived, not stored

The router needs the active leaf, the last leaf of each thread, and an activity
score per thread. All three are a pure function of the stored messages in
creation order, so they are recomputed from the tree on every route and nothing
new is persisted. Two processes sharing a store therefore agree on state as soon
as they agree on messages.

## Consequences

- `RoutingDecision.action` is `NEW | CONTINUE | SWITCH | FORK` with separate
  `confidence` (`FIRM | TENTATIVE`) and `decided_by` (`rules | explicit`). This
  replaced `NEW | ATTACH_FIRM | ATTACH_TENTATIVE` (breaking change, 0.4.0).
- Stored trees from 0.3.x load unchanged; old decisions keep their old fields.
- The decision is stored on the user node (`routing`), so the viewer can show,
  for every message, why it was placed where it is.
