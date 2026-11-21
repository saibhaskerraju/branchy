"""Score the router against labelled conversations (dev-time only).

    python eval/run.py                      # every file in eval/scenarios/
    python eval/run.py --verbose            # print each miss
    python eval/run.py --set tau_attach=0.4 # try a RouterConfig override
    python eval/run.py --llm                # + RoutingJudge on TENTATIVE (ADR 0004)

--llm reads LLM_MODEL / LLM_API_KEY / LLM_API_BASE and caches every answer in
eval/llm_cache/<model>.json, so a rerun is offline and reproducible; delete
the file to ask the model again.

Teacher forcing: each turn is routed on the tree built from the *gold*
placements so far, so every decision is judged on its own and one mistake
does not cascade.

Labels: each turn's "gold" is "NEW" or the id of the turn whose assistant
reply should be the parent.

Metrics
  exact            chosen parent == gold parent
  context loss     the gold parent is not on the chosen path to root, i.e. the
                   LLM would miss a message it needed (primary metric)
  NEW precision/recall, FORK precision
  errors by confidence: FIRM vs TENTATIVE
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from branchy import ConversationTree, RouterConfig, ThreadRouter  # noqa: E402
from branchy.router import RoutingJudge  # noqa: E402


def load(paths: list[Path]) -> list[dict]:
    convs = []
    for p in paths:
        convs.extend(json.loads(p.read_text("utf-8")))
    return convs


def cached_judge(kinds: list[str]) -> tuple[RoutingJudge, callable]:
    """A RoutingJudge on the env model, with answers cached on disk."""
    model = os.environ["LLM_MODEL"]
    live = RoutingJudge.from_litellm(
        model, os.environ.get("LLM_API_KEY"), os.environ.get("LLM_API_BASE") or None
    )
    path = ROOT / "eval" / "llm_cache" / (re.sub(r"[^A-Za-z0-9._-]", "_", model) + ".json")
    cache = json.loads(path.read_text("utf-8")) if path.exists() else {}

    def complete(messages: list[dict]) -> str:
        key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if key not in cache:
            cache[key] = live.complete(messages)
        return cache[key]

    def save() -> None:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(cache, indent=1, sort_keys=True), "utf-8")

    return RoutingJudge(complete, kinds=kinds, model_name=model), save


def run(convs: list[dict], router: ThreadRouter, verbose: bool,
        judge: RoutingJudge | None = None) -> dict:
    stats = dict(turns=0, exact=0, accepted=0, loss=0, new_tp=0, new_fp=0, new_fn=0,
                 fork_pred=0, fork_ok=0, err_firm=0, err_tent=0, tent=0,
                 llm=0, llm_fixed=0, llm_broke=0, llm_moved_still_wrong=0, llm_failed=0)
    by_kind: dict[str, list[int]] = {}  # uncertainty -> [tentative, errors]
    stats["by_kind"] = by_kind
    for conv in convs:
        tree = ConversationTree()
        reply_of: dict[str, str] = {}   # turn id -> assistant message id
        turn_of: dict[str, str] = {}    # assistant message id -> turn id
        for turn in conv["turns"]:
            gold = turn["gold"]
            gold_parent = None if gold == "NEW" else reply_of[gold]
            # optional "accept": other turns whose reply is also a right parent
            # (e.g. the earlier answer a "mentioned previously" points at)
            accept = {reply_of[a] for a in turn.get("accept", [])}
            d = router.route(tree, turn["user"])
            if judge is not None:
                rules_ok = d.target_message_id == gold_parent or d.target_message_id in accept
                d = judge.review(tree, turn["user"], d, router)
                if d.llm is not None:
                    stats["llm"] += 1
                    stats["llm_failed"] += bool(d.llm.error)
                    if not d.llm.agreed:
                        now_ok = d.target_message_id == gold_parent or d.target_message_id in accept
                        key = ("llm_fixed" if now_ok else
                               "llm_broke" if rules_ok else "llm_moved_still_wrong")
                        stats[key] += 1
            pred = d.target_message_id
            stats["turns"] += 1
            stats["tent"] += d.tentative
            accepted = pred is not None and pred in accept
            ok = pred == gold_parent or accepted
            stats["accepted"] += accepted
            if d.tentative:
                k = by_kind.setdefault(d.uncertainty or "?", [0, 0])
                k[0] += 1
                k[1] += not ok
            stats["exact"] += ok
            if gold_parent is not None:
                path = {n.message_id for n in tree.get_path_to_root(pred)} if pred else set()
                stats["loss"] += gold_parent not in path and not accepted
            if gold == "NEW":
                stats["new_tp" if pred is None else "new_fn"] += 1
            elif pred is None:
                stats["new_fp"] += 1
            if d.action == "FORK":
                stats["fork_pred"] += 1
                stats["fork_ok"] += ok
            if not ok:
                stats["err_tent" if d.tentative else "err_firm"] += 1
                if verbose:
                    got = "NEW" if pred is None else turn_of.get(pred, "?")
                    print(f"  MISS {conv['id']}/{turn['id']}: gold={gold} got={got} "
                          f"{d.action} {d.confidence} | {turn['user']}\n       {d.reason}")
            uid = tree.add_message(turn["user"], turn["assistant"], parent_message_id=gold_parent)
            aid = tree.messages[uid].child_message_ids[0]
            reply_of[turn["id"]] = aid
            turn_of[aid] = turn["id"]
    return stats


def report(s: dict) -> None:
    def pct(a, b):
        return f"{100 * a / b:5.1f}%" if b else "  n/a"

    errors = s["turns"] - s["exact"]
    print(f"turns              {s['turns']}")
    print(f"exact parent       {pct(s['exact'], s['turns'])}  ({s['exact']}/{s['turns']})")
    if s["accepted"]:
        print(f"  of which accepted alternatives  {s['accepted']}")
    attach_turns = s["turns"] - s["new_tp"] - s["new_fn"]
    print(f"context loss       {pct(s['loss'], attach_turns)}  ({s['loss']}/{attach_turns} attach turns)")
    print(f"NEW precision      {pct(s['new_tp'], s['new_tp'] + s['new_fp'])}")
    print(f"NEW recall         {pct(s['new_tp'], s['new_tp'] + s['new_fn'])}")
    print(f"FORK precision     {pct(s['fork_ok'], s['fork_pred'])}  ({s['fork_ok']}/{s['fork_pred']})")
    print(f"tentative share    {pct(s['tent'], s['turns'])}")
    print(f"errors FIRM / TENT {s['err_firm']} / {s['err_tent']}  (of {errors})")
    for kind, (n, err) in sorted(s["by_kind"].items()):
        print(f"  tentative {kind:9} {n:3} decisions, {err} wrong")
    if s["llm"]:
        print(f"llm reviewed       {s['llm']}  (failed {s['llm_failed']})")
        print(f"llm overrides      fixed {s['llm_fixed']}, broke {s['llm_broke']}, "
              f"moved but still wrong {s['llm_moved_still_wrong']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--llm", action="store_true", help="review TENTATIVE with an LLM")
    ap.add_argument("--llm-kinds", default="band,reference,margin")
    args = ap.parse_args()
    files = args.files or sorted((ROOT / "eval" / "scenarios").glob("*.json"))
    overrides = {}
    fields = {f.name: f.type for f in dataclasses.fields(RouterConfig)}
    for kv in args.set:
        key, value = kv.split("=", 1)
        if key not in fields:
            raise SystemExit(f"unknown RouterConfig field: {key}")
        overrides[key] = float(value) if "." in value else int(value)
    router = ThreadRouter(RouterConfig(**overrides))
    convs = load(files)
    print(f"{len(convs)} conversations from {', '.join(f.name for f in files)}")
    judge, save = cached_judge(args.llm_kinds.split(",")) if args.llm else (None, None)
    try:
        report(run(convs, router, args.verbose, judge))
    finally:
        if save:
            save()


if __name__ == "__main__":
    main()
