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

# Display the tree structure
print(chat.show_tree())

# Display a subtree starting from a specific message
print(chat.show_subtree(message_id_1))
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
- `BranchingChat.show_tree() -> str`
: Renders user messages in an ASCII tree format and returns the formatted string.
- `BranchingChat.show_subtree(message_id: str) -> str`
: Renders a subtree starting from a given `message_id` and all its child branches.
