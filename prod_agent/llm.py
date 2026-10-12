"""Live LLM wiring for the demo (OpenAI-compatible)."""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def make_llm(model: str | None = None, base_url: str | None = None,
             api_key: str | None = None, temperature: float = 0.2):
    key = api_key or os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set (see .env.example)")
    from openai import OpenAI

    client = OpenAI(base_url=base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
                    api_key=key)
    resolved = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    def llm(messages):
        resp = client.chat.completions.create(
            model=resolved, messages=messages, temperature=temperature)
        return resp.choices[0].message.content or ""

    return llm
