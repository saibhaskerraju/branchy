"""Import an existing transcript, persist it, reopen it, and pull a branch.

What it shows: add_message(..., assistant_message=...) to store turns without
an LLM; SQLite persistence across processes; get_path_to_root for branch
history; static HTML export.
Run:  python examples/import_history.py [--db PATH]   (run it twice)
Needs LLM: no.
"""

from __future__ import annotations

import argparse
import logging

from _common import SAMPLE_TURNS, describe

from branchy import BranchingChat

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="examples/.branchy/import.sqlite3")
    args = ap.parse_args()

    chat = BranchingChat(responder=lambda h, m: "", store=args.db, tree_id="import")

    if chat.tree.messages:
        print(f"tree already has {len(chat.tree.messages)} nodes; skipping replay")
    else:
        for user, assistant in SAMPLE_TURNS:
            chat.add_message(user, assistant_message=assistant)
            print(f"{user!r:50s} {describe(chat.last_decision)}")

    print("\n" + chat.show_tree())

    # Reopen in a fresh instance: everything comes back from SQLite.
    again = BranchingChat(responder=lambda h, m: "", store=args.db, tree_id="import")
    assert again.show_tree() == chat.show_tree()
    print(f"\nreopened: {len(again.tree.messages)} nodes, render identical")

    # Pull one branch: root -> last user message, as the LLM would see it.
    last_user = max(
        (n for n in again.tree.messages.values() if n.node_type == "user"),
        key=lambda n: n.created_seq,
    )
    print(f"\nbranch for {last_user.content!r}:")
    for node in again.tree.get_path_to_root(last_user.message_id):
        print(f"  {node.node_type:9s} {node.content}")

    again.export_viewer_html("examples/out/import_tree.html")
    print("\nstatic viewer written to examples/out/import_tree.html")
    chat.close()
    again.close()


if __name__ == "__main__":
    main()
