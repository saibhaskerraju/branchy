from __future__ import annotations

import pytest

from branchy import ConversationTree, ThreadRouter


class Script:
    """Feed (user, canned assistant reply) turns through the router and the
    tree, the same way BranchingChat does, without an LLM or a store."""

    def __init__(self, router: ThreadRouter | None = None) -> None:
        self.router = router or ThreadRouter()
        self.tree = ConversationTree()
        self.users: list[str] = []       # user message ids, in order
        self.assistants: list[str] = []  # their replies
        self.decisions = []

    def say(self, user: str, assistant: str, **kwargs):
        decision = self.router.route(self.tree, user, **kwargs)
        uid = self.tree.add_message(
            user, assistant,
            parent_message_id=decision.target_message_id,
            tentative=decision.tentative,
            routing=decision.model_dump(),
        )
        self.users.append(uid)
        self.assistants.append(self.tree.messages[uid].child_message_ids[0])
        self.decisions.append(decision)
        return decision

    def A(self, n: int) -> str:
        """Assistant message of turn n (1-based)."""
        return self.assistants[n - 1]

    def U(self, n: int) -> str:
        return self.users[n - 1]

    def parent_of_turn(self, n: int):
        return self.tree.messages[self.U(n)].parent_message_id


@pytest.fixture(scope="session")
def router() -> ThreadRouter:
    return ThreadRouter()


@pytest.fixture
def script(router) -> Script:
    return Script(router)


class FakeLLM:
    """Stands in for litellm.completion: replies come from `replies` (or a
    stub), routing reviews get `judge_answer`. Every call is recorded."""

    def __init__(self) -> None:
        self.replies: dict[str, str] = {}
        self.judge_answer = ""  # "" = unparseable -> the rules' decision stays
        self.reply_calls: list[list[dict]] = []
        self.judge_calls: list[list[dict]] = []

    def __call__(self, **kwargs):
        from types import SimpleNamespace

        from branchy.router.judge import SYSTEM_PROMPT

        messages = kwargs["messages"]
        if messages[0]["content"] == SYSTEM_PROMPT:
            self.judge_calls.append(messages)
            text = self.judge_answer
        else:
            self.reply_calls.append(messages)
            user = messages[-1]["content"]
            text = self.replies.get(user, f"(stub) {user}")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def histories(self) -> list[list[dict]]:
        """The history each reply call got (system prompt and new message removed)."""
        return [m[1:-1] for m in self.reply_calls]


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    import litellm

    fake = FakeLLM()
    monkeypatch.setattr(litellm, "completion", fake)
    return fake
