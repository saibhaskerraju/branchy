"""Console chat with a persisted, branching history and the live tree viewer.

What it shows: BranchingChat in a plain script — SQLite store, standalone
viewer, routing summary per turn, /tree and /why commands.
Run:  python examples/console_chat.py [--offline] [--user NAME]
      (type messages; /tree, /why, /quit)
Needs LLM: yes, unless --offline (set LLM_MODEL, LLM_API_KEY, LLM_API_BASE).
"""

from __future__ import annotations

import argparse
import logging
import sys

from _common import describe, llm_kwargs, stub_responder

from branchy import BranchingChat

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="stub replies, no LLM")
    ap.add_argument("--user", default="console", help="tree id (one tree per user)")
    args = ap.parse_args()

    kwargs = {"responder": stub_responder} if args.offline else llm_kwargs()
    chat = BranchingChat(
        store="examples/.branchy/console.sqlite3",
        tree_id=args.user,
        viewer=True,  # standalone server; URL is logged at INFO
        **kwargs,
    )
    print(f"viewer: {chat.viewer_url}   commands: /tree /why /quit")

    try:
        while True:
            try:
                line = input("> ") if sys.stdin.isatty() else sys.stdin.readline()
            except EOFError:
                break
            if not line:
                break
            line = line.strip()
            if line in ("/quit", "/exit"):
                break
            if line == "/tree":
                print(chat.show_tree())
                continue
            if line == "/why":
                d = chat.last_decision
                if d is None:
                    print("no routing yet")
                    continue
                print(d.reason)
                for c in d.candidates:
                    print(f"  {c.node_id[:8]} {c.kind:4s} bm25={c.bm25:.2f} rec={c.recency:.2f} "
                          f"s1={c.s1:.2f} evidence={c.evidence}"
                          + ("  (dominated root)" if c.node_id in d.dominated_root_ids else ""))
                continue
            if not line:
                continue
            message_id = chat.add_message(line)
            reply = chat.tree.messages[chat.tree.messages[message_id].child_message_ids[0]].content
            print(f"assistant: {reply}")
            print(f"           {describe(chat.last_decision)}")
    finally:
        chat.close()


if __name__ == "__main__":
    main()
