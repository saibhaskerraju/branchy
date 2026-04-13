# branchy-chat

Store conversation messages as a tree structure, retrieve relevant context traces, and let your LLM automatically route new prompts to the right branch.

- 🌳 Tree-based conversation memory
- 🔍 Retrieve relevant conversation traces
- 🧠 LLM-powered branch routing
- 📦 Publishable pip package

## Installation

```bash
pip install branchy-chat
```

## Usage

```python
from branchy import BranchingChat

chat = BranchingChat(
	model="openai/gpt-4o-mini",
	api_key="<your-api-key>",
	api_base="http://localhost:4000",  # Optional, useful for LiteLLM proxy
)

message_id_1 = chat.add_message("How do I build an index in Postgres?")
message_id_2 = chat.add_message("What if I need it concurrently?")

# Prints and returns the tree as text.
chat.show_tree()
```

### LiteLLM compatibility

`BranchingChat` supports LiteLLM-style parameters:

- `model`
- `api_key`
- `api_base`

This means you can point it at a LiteLLM proxy endpoint by setting `api_base`.

## API

- `BranchingChat(*, model, api_key=None, api_base=None, **lm_kwargs)`
: Initialize using LiteLLM-compatible arguments.
- `BranchingChat.add_message(user_message: str) -> str`
: Routes a message to the best branch, stores it in the tree, and returns the new `message_id`.
- `BranchingChat.show_tree(print_output: bool = True) -> str`
: Renders user messages in an ASCII tree format and prints it by default.

## Release Automation (GitHub Actions)

This repository includes two workflows:

1. `.github/workflows/create-release.yml`
: Manual workflow that validates the version, creates tag `vX.Y.Z`, and publishes a GitHub Release.
2. `.github/workflows/publish-pypi.yml`
: Automatically publishes to PyPI when a GitHub Release is published.

One-time setup:

1. Create a PyPI API token in your PyPI account.
2. In GitHub, open repository settings and add a secret named `PYPI_API_TOKEN`.
3. Run the `Create Release Tag` workflow from Actions with input `version` matching `pyproject.toml`.

After release publish, the PyPI workflow will:

1. Build the package.
2. Run `twine check`.
3. Upload `dist/*` to PyPI.
