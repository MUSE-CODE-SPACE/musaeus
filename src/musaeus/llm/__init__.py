"""llm/ — one interface to talk to any model, local or cloud.

The whole point: the rest of Musaeus never cares *which* provider is behind the
model. It calls `chat(messages, tools=...)` and gets back a normalized `Reply`
with either text or tool calls. Swapping Gemma-on-your-laptop for cloud Claude is
a one-line config change, not a rewrite.

We use a single `httpx` client for every provider (no vendor SDK required), so you
can read exactly what each API expects:
  - local + openai + google -> the OpenAI-compatible /chat/completions shape.
    Ollama serves it at :11434/v1; Gemini serves it at
    generativelanguage.googleapis.com/v1beta/openai.
  - anthropic               -> the native /v1/messages shape.

One dialect inside, many dialects outside: the rest of Musaeus (the agent, the
extractor) always speaks the OpenAI-compatible message shape, and THIS module
owns every translation out of it. That's why `_anthropic` rewrites assistant
`tool_calls` and `tool`-role messages into Anthropic content blocks — the
normalization lives in exactly one place, and everything above it stays
provider-blind.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Settings

Message = dict[str, Any]  # {"role": "user"|"assistant"|"system"|"tool", "content": ...}

# Frontier reasoning models have locked the sampling knobs: Anthropic removed
# `temperature` on Claude 4.7 and later (adaptive thinking replaced it — sending
# it is a 400), and OpenAI's GPT-5 family accepts only the default. The lesson in
# `sampling.py` still applies everywhere else (local models, Gemini, older cloud
# models); for these models we simply stop sending the knob.
_NO_SAMPLING_PREFIXES = (
    "claude-fable-5", "claude-opus-5", "claude-sonnet-5",
    "claude-opus-4-7", "claude-opus-4-8",
    "gpt-5", "o3", "o4",
)


def _supports_sampling(model: str) -> bool:
    return not model.startswith(_NO_SAMPLING_PREFIXES)


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
        max_tokens: int = 4096,
    ) -> Reply:
        # 4096 by default, not a few hundred: reasoning models spend "thinking"
        # tokens inside this same cap, so a small cap truncates mid-answer.
        if self.s.provider == "anthropic":
            return self._anthropic(messages, tools, temperature, max_tokens)
        return self._openai_compat(messages, tools, temperature, max_tokens)

    # -- OpenAI-compatible path (local Ollama, OpenAI, Gemini) ---------------
    def _openai_compat(self, messages, tools, temperature, max_tokens) -> Reply:
        if self.s.provider == "local":
            base, key = self.s.local_base_url, "ollama"
        elif self.s.provider == "google":
            # Gemini's OpenAI-compatibility endpoint: same wire shape, Google key.
            base, key = "https://generativelanguage.googleapis.com/v1beta/openai", \
                        self.s.google_api_key or ""
        else:
            base, key = "https://api.openai.com/v1", self.s.openai_api_key or ""

        # The GPT-5 era renamed the output cap: OpenAI rejects `max_tokens` on
        # current models and requires `max_completion_tokens` (the cap now covers
        # reasoning tokens too). Ollama and Gemini's compatibility gateways still
        # speak the original name, so the key depends on the provider.
        token_key = "max_completion_tokens" if self.s.provider == "openai" else "max_tokens"

        body: dict[str, Any] = {"model": self.model, "messages": messages, token_key: max_tokens}
        if _supports_sampling(self.model):
            body["temperature"] = temperature
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
        system, convo = _to_anthropic_messages(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": convo,
        }
        if system:
            body["system"] = system
        if _supports_sampling(self.model):
            body["temperature"] = temperature
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
        # Current models may also emit "thinking" blocks; we keep only what the
        # caller can act on (text + tool_use) and leave the rest in `raw`.
        for block in data.get("content", []):
            if block["type"] == "text":
                text += block["text"]
            elif block["type"] == "tool_use":
                calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block["input"]))
        return Reply(text=text, tool_calls=calls, raw=data)


def _to_anthropic_messages(messages: list[Message]) -> tuple[str, list[Message]]:
    """Translate our internal (OpenAI-shaped) history into Anthropic's dialect.

    Three differences, all handled here so callers never think about them:

    1. `system` is a top-level field, not a message role — we collect and join
       any system messages.
    2. An assistant turn that requested tools carries `tool_calls`; Anthropic
       expects the same information as `tool_use` *content blocks* on the
       assistant message.
    3. Tool results are not a `tool` role; they are `tool_result` content blocks
       inside a *user* message — and every result answering one assistant turn
       must ride in a single user message, so consecutive results are merged.
    """
    system_parts: list[str] = []
    convo: list[Message] = []

    for m in messages:
        role = m.get("role")

        if role == "system":
            system_parts.append(str(m.get("content", "")))
            continue

        if role == "assistant" and m.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m["tool_calls"]:
                fn = c.get("function", {})
                args = fn.get("arguments")
                # On the OpenAI wire, arguments are a JSON *string*; Anthropic
                # wants the parsed object.
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                blocks.append({
                    "type": "tool_use", "id": c["id"],
                    "name": fn.get("name", ""), "input": args or {},
                })
            convo.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": str(m.get("content", "")),
            }
            prev = convo[-1] if convo else None
            if (
                prev is not None
                and prev["role"] == "user"
                and isinstance(prev["content"], list)
                and prev["content"]
                and prev["content"][0].get("type") == "tool_result"
            ):
                prev["content"].append(block)  # merge into the same user turn
            else:
                convo.append({"role": "user", "content": [block]})
            continue

        convo.append(m)

    return "\n".join(p for p in system_parts if p), convo


def build_llm(settings: Settings) -> LLM:
    return LLM(settings)
