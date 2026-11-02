# branchy-chat

Store conversation messages as a tree structure, retrieve relevant context traces, and route new prompts to the right branch deterministically.

- 🌳 Tree-based conversation memory
- 🔍 Deterministic thread routing (BM25 + recency, no model required)
- 🧠 Any LLM via [LiteLLM](https://docs.litellm.ai/) for replies (or your own callable)
- 🖥️ Built-in HTML tree viewer
- 📦 Publishable pip package — runtime deps are just pydantic + litellm

## Installation

```bash
pip install branchy-chat
```

## Examples

Runnable, self-contained scripts in [`examples/`](examples/). All of them accept
`--offline` / `OFFLINE=1` (stub replies, no API key); otherwise set
`LLM_MODEL`, `LLM_API_KEY` and optionally `LLM_API_BASE`.

| File | Shows | Run |
|---|---|---|
| [`console_chat.py`](examples/console_chat.py) | Console assistant: persisted branching history, live viewer, `/tree` and `/why` | `python examples/console_chat.py --offline` |
| [`fastapi_app.py`](examples/fastapi_app.py) | FastAPI: `/chat`, `/history/{id}`, viewer mounted at `/branchy`, safe with `--workers 2` | `OFFLINE=1 uvicorn examples.fastapi_app:app --port 8080` |
| [`import_history.py`](examples/import_history.py) | Import a transcript without an LLM, reopen it from SQLite, pull one branch, export static HTML | `python examples/import_history.py` |
| [`router_only.py`](examples/router_only.py) | `ThreadRouter` + `ConversationTree` alone: full decision/candidate dump, config override, determinism | `python examples/router_only.py` |
| [`route_demo.py`](examples/route_demo.py) | Routing walkthrough over three scripted scenarios | `python examples/route_demo.py` |

## Usage

```python
from branchy import BranchingChat

chat = BranchingChat(model="gpt-4o-mini", api_key="<your-api-key>")

message_id_1 = chat.add_message("How do I build an index in Postgres?")
message_id_2 = chat.add_message("What if I need it concurrently?")

print(chat.show_tree())
print(chat.show_subtree(message_id_1))
```

Any provider LiteLLM supports works — pass the provider-prefixed model string:

```python
chat = BranchingChat(model="anthropic/claude-3-5-haiku-latest", api_key="<key>")
```

For custom endpoints (LiteLLM proxy, Ollama, vLLM), set `api_base` and use the
prefix your endpoint expects (e.g. `model="openai/my-model"` against a LiteLLM
proxy).

### Bring your own responder

Pass any callable `(history, user_message) -> reply`, where `history` is a list
of `{"role", "content"}` turns:

```python
chat = BranchingChat(responder=lambda history, msg: my_llm(history, msg))
```

## How routing works

Every message is placed by a deterministic, LLM-free router before the LLM is
called: it returns `NEW`, `ATTACH_FIRM` or `ATTACH_TENTATIVE` plus the target
node, and `chat.last_decision` carries the full candidate table and reason.
The algorithm, its thresholds and the reasoning behind it are specified in
[ADR 0001](decisions/0001-deterministic-thread-router.md).

## Design decisions

Architecture decision records live in [`decisions/`](decisions/):

| ADR | Topic |
|---|---|
| [0001](decisions/0001-deterministic-thread-router.md) | Deterministic lexical thread router ("sorting hat") |

## Tree viewer

A zero-dependency HTML view of the whole forest: every stored node as a box
(user and assistant separately), collapsible branches, badges for `tentative`,
`fork`, `active` and `root`, and a detail pane on click with the full prompt /
response, node metadata, the routing decision and candidate scores that placed
it, the tower sticker, the path to root and the exact history the LLM received.
It auto-refreshes (polls with `ETag`/`304`, re-renders only when the tree
changed; a checkbox pauses it). It is a development / observability aid and has
**no authentication**.

There are two ways to serve it; pick the one that matches how your code runs.

### Inside a web app (FastAPI, Starlette, any ASGI or WSGI host) — recommended

Mount it on your own app. It shares your port, your middleware and your auth;
no extra thread, no extra socket, nothing to map in compose, and nothing that
can collide with workers.

```python
from fastapi import FastAPI
from branchy import BranchingChat

chat = BranchingChat(model="gpt-4o-mini", api_key="...")
app = FastAPI()
app.mount("/branchy", chat.viewer_app())        # open http://<your app>/branchy/
# WSGI hosts (Flask, Django): chat.viewer_app(kind="wsgi")
```

Mount it behind whatever authentication protects the rest of your app. See
[`examples/fastapi_app.py`](examples/fastapi_app.py) for a runnable version.

Running several worker processes (`uvicorn --workers N`, gunicorn)? With the
default SQLite store every worker shares the same file, so the viewer — which
reads the store, not an in-memory tree — shows one consistent set of trees no
matter which worker answers. With `store=None` each worker holds its own
in-memory tree instead.

### Standalone (scripts, notebooks, console apps)

When there is no web app to mount into, `branchy` can run a tiny stdlib HTTP
server in a daemon thread of your process:

```python
chat = BranchingChat(model="gpt-4o-mini", api_key="...", viewer=True)
# INFO  branchy.viewer: branchy viewer: http://127.0.0.1:8787
chat.viewer_url                        # None when not running
chat.stop_viewer() / chat.close()      # or use `with BranchingChat(...) as chat:`
```

Options: `viewer_port` (default `8787`; `0` lets the OS pick), `viewer_host`
(default `127.0.0.1`), `viewer_refresh_seconds` (default `2`). The viewer can
never break your program: a busy port falls back to a free one with a `WARNING`
naming the real URL; if nothing can be bound it logs a warning and stays off.
The server stops when your process exits.

Inside a container (compose, Kubernetes) a loopback-bound viewer is not
reachable from your host browser. `branchy` detects this and logs a `WARNING`
telling you so; either mount it into your web app as above, or pass
`viewer_host="0.0.0.0"` and publish the port (`ports: ["8787:8787"]`).

### Static export

`chat.export_viewer_html("tree.html")` writes a single self-contained HTML file
with the current snapshot inlined — no server, no refresh — for sharing or
archiving.

### Storage-agnostic

The viewer only reads a snapshot provider (a JSON dict: one schema-1 tree
snapshot or `{"schema":1,"trees":[...]}`), so `asgi_app(provider)` /
`wsgi_app(...)` / `TreeViewer(...)` accept any callable returning that shape.
With a store configured, `chat.viewer_app()` / `chat.start_viewer()` serve
`store.snapshot_all` — every `tree_id` grouped under its own header.

An external viewer can also read a db file directly — e.g. on a mounted
volume while the app runs in Docker:

```bash
python -m branchy.viewer --db /path/to/branchy.sqlite3 --port 8787
```

`viewer_policy` controls the standalone server: `"auto"` (default) suppresses
it when a web-server process is detected (mount `chat.viewer_app()` instead),
`"allow"` starts regardless, `"deny"` never starts. `BRANCHY_VIEWER_DISABLE=1`
is a hard kill switch — the only env var the library reads: standalone never
starts and mounted apps answer 404.

## Persistence

By default every turn is persisted to `.branchy/branchy.sqlite3` (SQLite, WAL
mode — created automatically). Reopening a `BranchingChat` reloads the tree;
processes sharing the file stay consistent: `add_message` re-syncs the
in-memory tree when the file moved ahead, and each turn is written atomically
guarded by an expected version — a losing writer reloads, re-routes, and
retries once; a second conflict raises `StoreConflict`.

```python
chat = BranchingChat(..., store="path/to/db.sqlite3", tree_id="support")
chat = BranchingChat(..., store=None)   # in-memory only, writes nothing
```

In Docker, put the db on a volume (e.g. `-v branchy-data:/data`,
`store="/data/branchy.sqlite3"`) and view it from the host with
`python -m branchy.viewer --db /path/to/branchy.sqlite3`.

## API

- `BranchingChat(*, model=None, api_key=None, api_base=None, responder=None, router_config=None, system_prompt=None, store="default", tree_id="default", viewer=False, viewer_policy="auto", viewer_host="127.0.0.1", viewer_port=8787, viewer_refresh_seconds=2)`
- `BranchingChat.add_message(user_message, assistant_message=None) -> str`
  : routes, generates (or takes) a reply, stores it, returns the new `message_id`.
- `BranchingChat.show_tree() -> str` / `show_subtree(message_id) -> str`
- `BranchingChat.last_decision`: the `RoutingDecision` for the last add.
- `BranchingChat.viewer_app(kind="asgi" | "wsgi")`: mountable viewer for your web app.
- `BranchingChat.start_viewer()` / `stop_viewer()` / `viewer_url` / `close()` (also a context manager).
- `BranchingChat.export_viewer_html(path)`: static self-contained HTML.
- `ThreadRouter(config)` / `RouterConfig` / `RoutingDecision`: the router on its own (see `examples/router_only.py`).
- `SQLiteStore(path)`, `TreeStore` protocol, `StoreConflict` for custom persistence.
- `ConversationTree.to_dict()` / `from_dict(data)` / `snapshot()` for snapshots.
- `branchy.asgi_app(snapshot_provider)`, `branchy.wsgi_app(snapshot_provider)`, `branchy.TreeViewer(snapshot_provider, host, port, refresh_seconds)`, `branchy.render_static_html(snapshot)`.
