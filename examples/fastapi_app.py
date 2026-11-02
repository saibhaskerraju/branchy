"""Minimal FastAPI integration with the viewer mounted on the app's own port.

What it shows: one BranchingChat per process on a shared SQLite store (safe
with --workers N), a /chat endpoint, branch history lookup, and the viewer at
/branchy reading the store (all trees) — no second port.
Run:  pip install fastapi uvicorn
      OFFLINE=1 uvicorn examples.fastapi_app:app --port 8080 [--workers 2]
      curl -X POST localhost:8080/chat -H 'content-type: application/json' -d '{"message":"hi"}'
      open http://127.0.0.1:8080/branchy/
Needs LLM: yes, unless OFFLINE=1.  Set ENABLE_VIEWER=0 to not mount the viewer.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from branchy import BranchingChat, SQLiteStore, asgi_app

from examples._common import llm_kwargs, stub_responder  # noqa: E402

logging.basicConfig(level=logging.INFO)

DB = os.environ.get("BRANCHY_DB", "examples/.branchy/api.sqlite3")
chat: BranchingChat | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat
    kwargs = {"responder": stub_responder} if os.environ.get("OFFLINE") == "1" else llm_kwargs()
    chat = BranchingChat(store=DB, tree_id="api", **kwargs)
    yield
    chat.close()


app = FastAPI(lifespan=lifespan)

if os.environ.get("ENABLE_VIEWER", "1") == "1":
    # Reads the store directly, so every worker (and every tree) shows the same picture.
    app.mount("/branchy", asgi_app(SQLiteStore(DB).snapshot_all))


class ChatIn(BaseModel):
    message: str


@app.post("/chat")
def post_chat(body: ChatIn):
    message_id = chat.add_message(body.message)
    node = chat.tree.messages[message_id]
    reply = chat.tree.messages[node.child_message_ids[0]].content
    return {"reply": reply, "action": chat.last_decision.action, "message_id": message_id}


@app.get("/tree")
def get_tree():
    return {"tree": chat.show_tree()}


@app.get("/history/{message_id}")
def get_history(message_id: str):
    if message_id not in chat.tree.messages:
        raise HTTPException(404, "unknown message_id")
    return [{"role": n.node_type, "content": n.content} for n in chat.tree.get_path_to_root(message_id)]


@app.get("/healthz")
def healthz():
    return {"ok": True}
