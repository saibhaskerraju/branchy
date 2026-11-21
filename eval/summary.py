"""One-table routing accuracy summary over every eval set (dev-time).

    python eval/summary.py           # rules only, offline
    python eval/summary.py --llm     # + LLM review of TENTATIVE (needs LLM_MODEL, LLM_API_KEY)

Runs eval/run.py's teacher-forced evaluation on each set separately and
prints one row per set. TopiOCQA sets are included when they have been built
(python tools/build_topiocqa_eval.py). Exit code 0; it reports, it does not
gate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from run import cached_judge, load, run  # noqa: E402

from branchy import ThreadRouter  # noqa: E402

SETS = [
    ("project set", ["handwritten.json", "live.json", "reference-and-rare-term.json"]),
    ("short answers", ["affirmation.json"]),
]
TOPIOCQA = [
    ("TopiOCQA chains", "chains.json"),
    ("TopiOCQA interleaved", "interleaved.json"),
]


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:5.1f}%" if b else "  n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="review TENTATIVE with an LLM")
    args = ap.parse_args()

    scen = ROOT / "eval" / "scenarios"
    topi = ROOT / "eval" / "scenarios-topiocqa"
    sets = [(name, [scen / f for f in files]) for name, files in SETS]
    missing = []
    for name, f in TOPIOCQA:
        (sets.append((name, [topi / f])) if (topi / f).exists() else missing.append(name))

    router = ThreadRouter()
    judge, save = cached_judge(["band", "reference", "margin"]) if args.llm else (None, None)
    head = (f"{'set':<22}{'turns':>7}{'correct':>9}{'lost':>8}{'NEW prec':>10}"
            f"{'NEW rec':>9}{'forks ok':>10}{'tent':>7}{'FIRM err':>10}")
    print(head)
    print("-" * len(head))
    start = time.time()
    try:
        for name, files in sets:
            s = run(load(files), router, verbose=False, judge=judge)
            attach = s["turns"] - s["new_tp"] - s["new_fn"]
            print(f"{name:<22}{s['turns']:>7}{pct(s['exact'], s['turns']):>9}"
                  f"{pct(s['loss'], attach):>8}"
                  f"{pct(s['new_tp'], s['new_tp'] + s['new_fp']):>10}"
                  f"{pct(s['new_tp'], s['new_tp'] + s['new_fn']):>9}"
                  f"{pct(s['fork_ok'], s['fork_pred']):>10}"
                  f"{pct(s['tent'], s['turns']):>7}{s['err_firm']:>10}")
            if judge is not None:
                print(f"{'':<22}llm reviewed {s['llm']}, fixed {s['llm_fixed']}, "
                      f"broke {s['llm_broke']}, failed {s['llm_failed']}")
    finally:
        if save:
            save()
    print("-" * len(head))
    print(f"correct = chosen parent is the labelled one; lost = the needed parent was not "
          f"in the history sent;\ntent = share of TENTATIVE placements; FIRM err = errors the "
          f"LLM review never sees. {time.time() - start:.0f}s"
          + (" (rules only)" if judge is None else " (rules + LLM review)"))
    for name in missing:
        print(f"skipped {name}: run `make topiocqa` first")


if __name__ == "__main__":
    main()
