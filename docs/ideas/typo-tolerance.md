# Idea: typo tolerance for topic words

- Status: Idea, not scheduled (recorded 2026-11-21)
- Affects: `src/branchy/router/router.py` (lexical score),
  [ADR 0002](../../decisions/0002-routing-algorithm.md) step 2
- Related: [cue-classifier.md](cue-classifier.md), which covers typos in cue
  phrases
- If adopted, this becomes an ADR.

## Problem

Word matching is exact after stemming. A misspelled topic word ("contianer"
for "container") matches nothing, and it also makes every score worse.

The lexical score is the rarity weight of the shared terms divided by the
rarity weight of all the message's terms
([router.py:349](../../src/branchy/router/router.py#L349)). A term missing
from `idf.json` gets `unknown_milli` = 9393, the maximum weight. So a typo
becomes the heaviest term in the message, matches nothing, and lowers `lex`
against every earlier reply, including the right one.

The meaning score gives little help. The tokenizer splits the typo into word
pieces whose vectors are not related to the intended word.

The effect: a misspelled follow-up can fall into the uncertain band, or below
`tau_new` and start a new topic. That is the expensive mistake: the LLM
receives no history. Short follow-ups are hit hardest, since one typo is a large
share of the message.

Misspelled filler words ("teh", "becuase") have the same effect: they are not
in `STOPWORDS`, so they become topic words with maximum weight.

## Proposal

Before scoring, correct an **unknown** term to a term that **already appears in
this conversation**, when the two are within a small edit distance.

| Term length | Correction allowed |
|---|---|
| under 5 letters | none |
| 5 to 7 letters | 1 edit |
| 8 or more letters | 2 edits |

- **Only unknown terms** (not in `idf.json` and not stop words) are corrected.
  Known words stay exact, so "index" never becomes "indent".
- **Only towards the conversation's own terms.** Not towards the `idf.json`
  vocabulary, which holds only 4,901 stems and misses many real words.
  "postgres" is itself unknown there; correcting against the vocabulary could
  turn `postgr` into `poster`. Correcting only towards words already said
  cannot link the message to a topic nobody mentioned.
- **Deterministic:** the closest match wins; ties are broken by a fixed order
  (for example, the earliest term in creation order).
- The corrected term is scored with its corrected form's weight. Record each
  correction in the decision so the viewer can show it.

## Rejected for now

- **Fuzzy matching between all words.** Real words one letter apart are
  common ("index"/"indent", "port"/"post"/"sort", "list"/"lost"). It would
  join unrelated topics, the failure ADR 0003 was built to avoid. More matches
  also mean more evidence for forks, which hide history.
- **Fuzzy cue phrases.** A missed cue already fails safe: a missed `RETURN`
  means no fork, and a missed `NEW_TOPIC` means attach. Typos in cues are
  covered by [cue-classifier.md](cue-classifier.md) through training examples.

## Cost

Only unknown terms are compared, and only against the conversation's distinct
terms, so the work stays small. Precompute the term set per tree, as node
features are cached today. Keep the current budget of about 30 ms per route on
a 500-node tree.

## How to judge it

The current eval set probably has no typos, so the gain is unproven. First,
add a scenario file with realistic typo versions of existing conversations,
written before the change. Then compare with `eval/run.py`:

- must not lower FORK precision or NEW precision;
- should lower context loss and raise NEW recall and exact-parent accuracy on
  the typo scenarios;
- must leave the existing scenarios unchanged.

## Origin

This idea came from a review of the router on 2026-11-21, while checking
whether the router tolerates typos. It does not: stemming only removes word
endings, and the lexical score uses exact matches.
