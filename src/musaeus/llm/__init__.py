"""llm/ — one interface to talk to any model, local or cloud.

The whole point: the rest of Musaeus never cares *which* provider is behind the
model. It calls `chat(messages, tools=...)` and gets back a normalized `Reply`
with either text or tool calls. Swapping Gemma-on-your-laptop for cloud Claude is
a one-line config change, not a rewrite.

We use a single `httpx` client for every provider (no vendor SDK required), so you
can read exactly what each API expects:
  - local + openai  -> the OpenAI-compatible /chat/completions shape
  - anthropic       -> the native /v1/messages shape
Ollama serves the OpenAI-compatible shape, so "local" and "openai" share a path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Settings

Message = dict[str, Any]  # {"role": "user"|"assistant"|"system"|"tool", "content": ...}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    """A normalized model reply — the same shape no matter the provider."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLM:
    def __init__(self, settings: Settings, timeout: float = 120.0):
        self.s = settings
        self.model = settings.model_for
        self._http = httpx.Client(timeout=timeout)

    # -- public interface ---------------------------------------------------
    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Reply:
        if self.s.provider == "anthropic":
            return self._anthropic(messages, tools, temperature, max_tokens)
        return self._openai_compat(messages, tools, temperature, max_tokens)

    # -- OpenAI-compatible path (local Ollama + OpenAI) ---------------------
    def _openai_compat(self, messages, tools, temperature, max_tokens) -> Reply:
        if self.s.provider == "local":
            base, key = self.s.local_base_url, "ollama"
        else:
            base, key = "https://api.openai.com/v1", self.s.openai_api_key or ""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        r = self._http.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
        )
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = [
            ToolCall(id=c["id"], name=c["function"]["name"],
                     arguments=json.loads(c["function"]["arguments"] or "{}"))
            for c in (msg.get("tool_calls") or [])
        ]
        return Reply(text=msg.get("content") or "", tool_calls=calls, raw=data)

    # -- Anthropic native path ----------------------------------------------
    def _anthropic(self, messages, tools, temperature, max_tokens) -> Reply:
        # Anthropic keeps `system` as a top-level field, not a message role.
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [
                {"name": t["name"], "description": t.get("description", ""),
                 "input_schema": t["parameters"]}
                for t in tools
            ]
        r = self._http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.s.anthropic_api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        r.raise_for_status()
        data = r.json()
        text, calls = "", []
        for block in data.get("content", []):
            if block["type"] == "text":
                text += block["text"]
            elif block["type"] == "tool_use":
                calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block["input"]))
        return Reply(text=text, tool_calls=calls, raw=data)


def build_llm(settings: Settings) -> LLM:
    return LLM(settings)
