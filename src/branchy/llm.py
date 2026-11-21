from __future__ import annotations

from typing import Optional

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class LiteLLMClient:
    """The one model a BranchingChat talks to, via litellm.completion — works
    with OpenAI, Anthropic, a LiteLLM proxy, Ollama, vLLM, etc. The model
    string is passed verbatim; provider prefixes (e.g. "anthropic/…" or
    "openai/…" for a custom api_base) follow litellm's rules.

    Used for the replies (reply) and for reviewing TENTATIVE routing
    decisions (complete; ADR 0004).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        **completion_kwargs,
    ) -> None:
        import litellm

        litellm.suppress_debug_info = True
        self._litellm = litellm
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.completion_kwargs = completion_kwargs

    def complete(self, messages: list[dict], **overrides) -> str:
        """Chat messages in, the model's text out."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **self.completion_kwargs,
            **overrides,
        }
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        response = self._litellm.completion(**kwargs)
        return response.choices[0].message.content or ""

    def reply(self, history: list[dict], user_message: str) -> str:
        """The assistant reply, given the routed branch's history as
        {"role", "content"} turns (empty for a new topic)."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return self.complete(messages)
