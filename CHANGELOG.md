# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). While the version is below 1.0,
minor versions may contain breaking changes.

## [0.4.0] - 2026-11-21

### Breaking changes

- `BranchingChat` now requires `model`. The `responder=` argument,
  `branchy.echo_responder` and `branchy.LiteLLMResponder` are removed: the chat's
  model writes the replies and also checks uncertain placements.
  `add_message(..., assistant_message=...)` still accepts a reply you generated
  yourself.

### Added

- Uncertain (`TENTATIVE`) placements are checked by the chat's model before the
  reply is written; `FIRM` placements never call it. If the call fails or times
  out, the router's placement is kept
  ([ADR 0004](decisions/0004-llm-review-of-tentative-decisions.md)).
- `RoutingDecision.uncertainty` (`"band"`, `"reference"` or `"margin"`) says why a
  placement is TENTATIVE; `RoutingDecision.llm` records the model's review;
  `decided_by` can be `"llm"`.
- The viewer shows `llm agreed`, `llm override` and `llm failed` badges, and an
  "LLM review" panel.
- `eval/run.py --llm` measures routing with the model check (answers are cached
  for reproducible reruns), and the report breaks TENTATIVE errors down by kind.
- New evaluation scenarios: `eval/scenarios/live.json` and
  `eval/scenarios/reference-and-rare-term.json`.
- `LICENSE` (Apache-2.0) and `NOTICE`.
- New cue `AFFIRMATION` and scenario set `eval/scenarios/affirmation.json`
  (short answers to an assistant's question).
- TopiOCQA evaluation sets: `tools/build_topiocqa_eval.py` downloads TopiOCQA
  (pinned sha256) and writes `eval/scenarios-topiocqa/chains.json` and
  `interleaved.json`. They are git-ignored (CC BY-NC-SA 4.0) and run
  separately. Turns may list `accept` alternatives in the scenario format.
- `eval/summary.py` and `make check`: unit tests plus one accuracy table over
  every eval set. `make eval-llm` adds the LLM review.
- `docs/ideas/`: recorded, unscheduled ideas (trained cue classifier, typo
  tolerance, untested scenarios).

### Changed

- A reference such as "in that" can move to another topic only if that topic
  matches on its own (not only better than the current one).
- An uncertain match outside the current topic joins that topic when they share
  a rare word (such as "fastapi"), instead of starting a new topic.
- "want", "need", "understand", "learn" and "know" mark a message as a request,
  so short messages like "I want to understand docker" can start a new topic.

### Fixed

- Personal and place pronouns (*he, she, his, her, their, there, here*) and
  pointers (*former, latter, above, mentioned*) count as references, so
  "where was he born?" no longer starts a new topic.
- A message with a return cue ("the first one", "going back to") that matches no
  earlier words is no longer placed as a new topic.
- A short reply ("yes", "sure, go ahead", "the refund please") right after an
  assistant question continues that branch as FIRM, without an LLM review.
- Together, these routing fixes raise correct placements on TopiOCQA (one topic
  per conversation) from 85.9% to 95.8% and cut lost history from 15.4% to
  4.5%; the project set is unchanged (94.0%).
- Viewer: auto-refresh and the Refresh button now redraw the tree when it
  changes (previously only the first load was drawn).
- Viewer: a browser tab left open across an app restart no longer stays stale.
- Topic summaries are rebuilt in one pass: about 4 ms instead of about 210 ms
  per message at 120 messages.
- Viewer: polling an unchanged tree no longer reloads it from the database.
