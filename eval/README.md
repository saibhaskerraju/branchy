# Router evaluation (dev-time; not part of the package)

```bash
python eval/run.py                       # all files in eval/scenarios/
python eval/run.py --verbose             # print every miss with the router's reason
python eval/run.py --set tau_attach=0.4  # try a RouterConfig override
python eval/run.py --llm                 # + RoutingJudge on TENTATIVE decisions (ADR 0004)
```

`--llm` uses `LLM_MODEL` / `LLM_API_KEY` / `LLM_API_BASE` and caches every
answer in `llm_cache/<model>.json`: reruns are offline and reproducible. Delete
the file to ask the model again. `--llm-kinds band,reference` limits which
TENTATIVE kinds are reviewed.

## Scenario files

| File | Source |
|---|---|
| `handwritten.json` | written with the router design; used to choose its defaults |
| `live.json` | a real session with a real model (2026-11-14), labelled by hand |
| `reference-and-rare-term.json` | the live failure patterns on other topics, plus controls; written alongside the ADR 0004 fixes |
| `affirmation.json` | short answers ("yes", "no thanks", "the refund please") to an assistant's question, in business-style flows; written before the `AFFIRMATION` cue |

## TopiOCQA scenarios (separate set)

`scenarios-topiocqa/` holds scenarios converted from
[TopiOCQA](https://huggingface.co/datasets/McGill-NLP/TopiOCQA) (dev split,
205 conversations). They are not run by default; pass them explicitly:

```bash
python tools/build_topiocqa_eval.py                              # download (pinned sha256) and convert
python eval/run.py eval/scenarios-topiocqa/chains.json
python eval/run.py eval/scenarios-topiocqa/interleaved.json
```

| File | What it tests |
|---|---|
| `chains.json` | 205 conversations, 2,514 turns. Each turn continues the previous one. TopiOCQA's topic changes are drill-downs ("which hormones are related to it?"), so a topic change is not labelled NEW. Tests that drill-downs are not split off |
| `interleaved.json` | 88 pairs of conversations with no shared topic, interleaved in blocks of 4 turns. Tests NEW for each conversation's first turn and SWITCH back at each later block. The switches are splice points, easier than natural ones |

Assistant replies are the TopiOCQA answer plus its rationale sentence. The
data is CC BY-NC-SA 4.0 (non-commercial, share-alike), so the generated JSON is
git-ignored and never shipped; the script recreates it.

## Labels

Each file is a JSON list of conversations. Each turn has a canned assistant reply
(routing reads replies, and canned ones keep the run offline and exact) and a
`gold` label: `"NEW"`, or the id of the turn whose assistant reply should be
the parent.

```json
{"id": "t3", "gold": "t1", "user": "What if I need it to not lock the table?",
 "assistant": "Use CREATE INDEX CONCURRENTLY; ..."}
```

A turn may also list `"accept": ["t2", ...]`: other turns whose reply is also a
right parent. The TopiOCQA converter uses it for questions that point back past
the previous turn ("the country mentioned previously"). Accepted alternatives
count as correct and are reported separately.

Label for what the LLM needs: the parent is the answer the message continues.
Use a fork label (an earlier answer that already has a reply) only when the user
goes back and the later turns are no longer relevant.

`scenarios/handwritten.json` was written together with the router design and
was used to choose its defaults — review the labels, and add conversations the
router has not seen before trusting the numbers.

## Method and metrics

Teacher forcing: each turn is routed on the tree built from the gold placements
so far, so one mistake does not cascade and every decision is judged alone.

| Metric | Meaning |
|---|---|
| exact parent | chosen parent == gold parent |
| context loss | the gold parent is not in the history sent to the LLM (primary) |
| NEW precision / recall | new-topic decisions |
| FORK precision | share of forks that were right |
| errors FIRM / TENT | where the mistakes are: if they cluster in TENTATIVE decisions, a fallback for those would pay off |
| tentative by kind | TENTATIVE decisions and errors per `uncertainty` (`band`, `reference`, `margin`) |
| llm overrides | with `--llm`: overrides that fixed an error, broke a right answer, or moved a wrong one elsewhere |
