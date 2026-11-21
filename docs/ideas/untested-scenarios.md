# Idea: scenarios the router has not been tested on

- Status: Idea, not scheduled (recorded 2026-11-21)
- Affects: `eval/scenarios/`, `eval/scenarios-topiocqa/`, possible new sets
- Related: [cue-classifier.md](cue-classifier.md), [typo-tolerance.md](typo-tolerance.md)

## Where things stand

Measured on 2026-11-21, rules only, after the pronoun, return-cue and
`AFFIRMATION` fixes:

| Set | Turns | Exact parent | Context loss | FIRM errors |
|---|---|---|---|---|
| original (handwritten, live, reference-and-rare-term) | 67 | 94.0% | 2.9% | 0 |
| affirmation | 21 | 95.2% (100% with LLM review) | 8.3% | 0 |
| TopiOCQA chains | 2,514 | 95.8% | 4.5% | 84 |
| TopiOCQA interleaved | 2,165 | 80.8% | 20.0% | 207 |

None of these sets covers the cases below, so the router's behaviour on them is
unknown or only predicted from reading the rules.

## Untested cases

### 1. Conversational replies beyond yes/no

Hedges, preferences and short questions without a pronoun or a yes/no opener:
"maybe", "not really", "I'd prefer afternoons", "what are my options?",
"do you have anything cheaper?", "how much?", "any other dates?".

- **Prediction:** replies with ≤ 3 topic words and no question word are
  `FRAGMENT` and continue. Short questions with a question word and no pronoun
  get no cue, are scored on shared words, and are likely to become a FIRM new
  topic. The same mechanism causes the 70 FIRM NEW errors in TopiOCQA chains
  ("who was arrested?", "when was the film released?").
- **Sources:** DialSeg711 / MultiWOZ and the Stanford dialogue set (booking
  dialogues where the agent asks questions), SuperDialseg (doc2dial support
  dialogues), plus a hand-written file. Check licences first.

### 2. Generic closing questions in LLM replies

LLM replies often end with "Is there anything else I can help with?" or "Would
you like more details?". The `answers_question` rule treats any `?` in the last
300 characters as a question.

- **Risk:** a short real topic change right after such a reply ("how do I reset
  my password?") is attached to the old branch. Today only `FRAGMENT` and
  `AFFIRMATION` messages get this FIRM treatment, so the exposure is limited;
  any widening of the rule (fix "A" in the review) makes it larger.
- **Source:** hand-written. Pair each closing question with both a real answer
  and a real topic change.

### 3. Short topic changes without a cue

New topics made of common words and no "btw" / "unrelated": "what's a good
movie?", "any tips for sleeping better?".

- **Why it matters:** these are the controls for a "standalone check" before
  NEW (fix "B" in the review). Without them, a fix can look better while it
  silently attaches real topic changes.
- **Source:** TIAGE (PersonaChat with human topic-shift labels; agreement 0.479,
  so indicative), plus a hand-written file.

### 4. Long, real LLM replies

Only `live.json` (7 turns) has real LLM replies. Every other set uses short
canned replies or TopiOCQA's answer plus one sentence.

- **Why it matters:** long replies contain many terms, so they match almost
  anything (`lex` has no length normalisation), and they repeat earlier content,
  which weakens fork evidence (README limitation).
- **Source:** regenerate the replies of an existing set with a real model,
  keep the gold labels, and cache them the way `eval/llm_cache/` does.

### 5. Typos

- **Why it matters:** an unknown word gets the maximum IDF weight and lowers
  every score. See [typo-tolerance.md](typo-tolerance.md).
- **Source:** copies of existing scenarios with realistic typos, written before
  any fix.

### 6. Non-English messages

Cue lists, stop words and the stemmer are English only. The embedding model is
English-focused.

- **Unknown:** whether non-English chats fall back to word overlap gracefully
  or start a new topic on almost every message.
- **Source:** small hand-written set per language that matters to users.

### 7. Very long conversations

The longest scenario has about 25 turns. Not tested: 100+ turns, many open
topics, and how the activity prior (half-life 4 messages) behaves when a topic
returns after a long gap.

- **Also measure:** routing time per message as the tree grows (ADR 0003 states
  about 30 ms on a 500-node tree).

### 8. Latency and LLM review load

After the fixes, about 35–37% of TopiOCQA turns are TENTATIVE and would get an
LLM review (about 1 s each before the reply). Not measured end to end, and not
measured with a slow or failing provider.

- **Also check:** how often the LLM review breaks a correct placement on larger
  samples (interleaved sample: fixed 25, broke 11).

### 9. Natural returns to an earlier topic

TopiOCQA interleaved returns are splice points, and the returning questions
are pronoun-heavy ("the latter place's past?"), which is unnatural after a
detour. Returns are right only 47.8% of the time (193 of 404), but real returns
are probably different.

- **Source:** hand-labelled real multi-topic chats (WildChat conversations
  with a detour and a return), or the NTM benchmark if its data is released
  under a usable licence.

## How to use this list

For each case: write the scenarios first, run the current router to get a
baseline, and only then change rules. Agree on acceptance criteria before any
change (no lost correct answers on existing sets, no new FIRM errors, an
agreed limit on NEW recall loss and on the TENTATIVE share).
