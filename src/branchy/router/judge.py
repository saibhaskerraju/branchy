"""LLM review of TENTATIVE routing decisions (ADR 0004).

The deterministic router decides every message. When it is unsure
(decision.tentative), and only then, the RoutingJudge asks the LLM to pick
among a small, closed set of placements built from the router's own scores:

  - each thread's latest answer (continue / switch), at most `max_threads`;
  - the router's own pick, if it is something else (e.g. a fork);
  - a new topic.

The LLM answers with one option letter. It cannot invent a placement, so it
can never create a fork the router did not propose (forks hide context; see
ADR 0001). Any failure (error, timeout, unparseable answer) keeps the
router's decision: routing never breaks the chat.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Iterable, Optional

from ..tree import ConversationTree
from .router import LLMReview, RoutingDecision, ThreadRouter, derive_state

# chat messages ([{"role", "content"}]) -> the model's text reply
Complete = Callable[[list[dict]], str]

_LETTERS = "ABCDEFGHIJKLM"
_NEW_LETTER = "N"

SYSTEM_PROMPT = (
    "You place a new chat message into an existing conversation. The "
    "conversation has separate threads; you see the latest exchange of each. "
    "Decide which option the new message continues, so the assistant gets "
    "the right history.\n"
    "- A pronoun or short follow-up ('it', 'that', 'in that', 'and in "
    "python?') usually continues the most recent exchange.\n"
    "- A message that names something only one thread discusses continues "
    "that thread, even if it is not the most recent.\n"
    f"- Choose {_NEW_LETTER} only if the message starts a topic none of the "
    "options are about. Sharing a common word is not enough to continue.\n"
    "Answer with the option letter only."
)


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


class RoutingJudge:
    """Ask an LLM to settle TENTATIVE routing decisions.

    complete: (messages) -> text. BranchingChat always builds one on its own
    model; RoutingJudge.from_litellm(...) is for the eval and direct use.
    kinds: which uncertainties to review ("band", "reference", "margin";
    see RoutingDecision.uncertainty). Default: all.
    """

    def __init__(
        self,
        complete: Complete,
        *,
        kinds: Iterable[str] = ("band", "reference", "margin"),
        max_threads: int = 3,
        snippet_chars: int = 300,
        model_name: str = "",
    ) -> None:
        self.complete = complete
        self.kinds = frozenset(kinds)
        self.max_threads = max_threads
        self.snippet_chars = snippet_chars
        self.model_name = model_name

    @classmethod
    def from_litellm(
        cls,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        *,
        timeout: float = 10.0,
        **judge_kwargs,
    ) -> "RoutingJudge":
        """A judge on its own litellm model (BranchingChat builds one on its
        reply model; this is for the eval and direct router use)."""
        from ..llm import LiteLLMClient

        client = LiteLLMClient(model, api_key=api_key, api_base=api_base)
        return cls(lambda messages: client.complete(messages, timeout=timeout),
                   model_name=model, **judge_kwargs)

    # ------------------------------------------------------------ options

    def options(
        self, tree: ConversationTree, decision: RoutingDecision, router: ThreadRouter
    ) -> list[Optional[str]]:
        """Placements offered, most recently active thread first; None = new topic."""
        state = derive_state(tree, router.config.activity_decay)
        ranked = [t.root_id for t in decision.threads]  # router's thread ranking
        by_recency = sorted(
            state.thread_leaf,
            key=lambda r: tree.messages[state.thread_leaf[r]].created_seq,
            reverse=True,
        )
        roots: list[str] = []
        for r in [state.active_thread, *ranked, *by_recency]:
            if r and r in state.thread_leaf and r not in roots:
                roots.append(r)
        roots = roots[: self.max_threads]
        opts: list[Optional[str]] = [state.thread_leaf[r] for r in roots]
        if decision.target_message_id and decision.target_message_id not in opts:
            opts.append(decision.target_message_id)  # e.g. the router's fork
        # most recent first, so "the most recent exchange" is always option A
        opts.sort(key=lambda t: tree.messages[t].created_seq, reverse=True)
        opts.append(None)
        return opts

    def _describe(self, tree: ConversationTree, target: str) -> str:
        path = tree.get_path_to_root(target)
        user = tree.messages[target].parent_message_id
        lines = []
        if path and path[0].message_id != user:
            lines.append(f'thread started with: "{_clip(path[0].content, 120)}"')
        if user in tree.messages:
            lines.append(f'user: "{_clip(tree.messages[user].content, 200)}"')
        lines.append(f'assistant: "{_clip(tree.messages[target].content, self.snippet_chars)}"')
        return "\n    ".join(lines)

    def prompt(
        self, tree: ConversationTree, message: str, opts: list[Optional[str]]
    ) -> list[dict]:
        parts = []
        for i, target in enumerate(opts):
            if target is None:
                parts.append(f"[{_NEW_LETTER}] a new topic, unrelated to the options above")
                continue
            tag = " (most recent exchange)" if i == 0 else ""
            parts.append(f"[{_LETTERS[i]}]{tag}\n    {self._describe(tree, target)}")
        user = "Options:\n" + "\n".join(parts) + f'\n\nNew message: "{_clip(message, 500)}"'
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]

    # ------------------------------------------------------------ review

    def should_review(self, decision: RoutingDecision) -> bool:
        return (decision.tentative and decision.decided_by == "rules"
                and decision.uncertainty in self.kinds)

    def review(
        self,
        tree: ConversationTree,
        message: str,
        decision: RoutingDecision,
        router: ThreadRouter,
    ) -> RoutingDecision:
        """The decision to use: the router's, or the LLM's pick among options."""
        if not self.should_review(decision):
            return decision
        review = LLMReview(router_action=decision.action,
                           router_target=decision.target_message_id,
                           model=self.model_name)
        started = time.monotonic()
        try:
            opts = self.options(tree, decision, router)
            review.options = opts
            answer = self.complete(self.prompt(tree, message, opts))
            review.answer = _clip(answer or "", 40)
            letter = _parse_letter(answer or "", len(opts) - 1)
            if letter is None:
                raise ValueError(f"no option letter in answer {review.answer!r}")
            review.choice = letter
            target = None if letter == _NEW_LETTER else opts[_LETTERS.index(letter)]
        except Exception as exc:  # any failure keeps the router's decision
            review.error = f"{type(exc).__name__}: {exc}"[:200]
            review.latency_ms = int((time.monotonic() - started) * 1000)
            return decision.model_copy(update={
                "llm": review,
                "reason": decision.reason + f" | llm failed, kept rules ({review.error})",
            })
        review.latency_ms = int((time.monotonic() - started) * 1000)
        review.agreed = target == decision.target_message_id
        if review.agreed:
            return decision.model_copy(update={
                "llm": review, "reason": decision.reason + " | llm agreed",
            })
        state = derive_state(tree, router.config.activity_decay)
        action = router._classify(tree, target, state)
        return decision.model_copy(update={
            "llm": review,
            "action": action,
            "target_message_id": target,
            "thread_id": state.root_of.get(target) if target else None,
            "decided_by": "llm",
            "reason": decision.reason
            + f" | llm chose {review.choice}: {action.lower()} (rules said "
              f"{decision.action.lower()})",
        })


def _parse_letter(answer: str, n_threads: int) -> Optional[str]:
    """The first standalone option letter in the answer, if valid."""
    valid = set(_LETTERS[:n_threads]) | {_NEW_LETTER}
    for m in re.finditer(r"\b([A-Z])\b", answer.strip().upper()):
        if m.group(1) in valid:
            return m.group(1)
    return None
