from __future__ import annotations

from typing import Callable, Optional

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# (history, user_message) -> assistant reply, where history is a list of
# {"role": "user"|"assistant", "content": str} turns (empty for a new thread).
Responder = Callable[[list[dict], str], str]


class LiteLLMResponder:
    """Responder backed by litellm.completion — works with OpenAI, Anthropic,
    a LiteLLM proxy, Ollama, vLLM, etc. The model string is passed verbatim;
    provider prefixes (e.g. "anthropic/…" or "openai/…" for a custom
    api_base) follow litellm's rules.
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

    def __call__(self, history: list[dict], user_message: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **self.completion_kwargs,
        }
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        response = self._litellm.completion(**kwargs)
        return response.choices[0].message.content
