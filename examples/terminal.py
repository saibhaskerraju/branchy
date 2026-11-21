"""Scratch terminal for trying branchy.  python examples/terminal.py
Commands: /tree  /why  /quit

Reads the model from the environment (set in .env.local):
LLM_MODEL, LLM_API_KEY, LLM_API_BASE (optional)."""

import json
import os

from branchy import BranchingChat

chat = BranchingChat(
    model=os.environ["LLM_MODEL"],
    api_key=os.environ["LLM_API_KEY"],
    api_base=os.environ.get("LLM_API_BASE") or None,
    viewer=True,               # HTML viewer on/off
    viewer_port=8787,          # viewer port
)

while (line := input("> ").strip()) != "/quit":
    if line == "/tree":
        print(chat.show_tree())
    elif line == "/why":
        print(chat.explain())
    elif line:
        chat.add_message(line)
        # the whole branch, from the root down to the new reply, as chat messages
        reply_id = chat.tree.messages[chat.last_message_id].child_message_ids[0]
        conversation = [
            {"role": node.node_type, "content": node.content}
            for node in chat.tree.get_path_to_root(reply_id)
        ]
        print(json.dumps(conversation, indent=2, ensure_ascii=False))
        print("  ", chat.describe())
chat.close()
