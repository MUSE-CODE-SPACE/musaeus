"""agent.py — the ReAct loop: how a model turns *thinking* into *doing*.

ReAct = **Rea**soning + **Act**ing. Instead of answering in one shot, the model
runs a loop:

    Thought       — the model reasons about what to do next (its text reply).
    Action        — it asks to call a tool (a structured tool call).
    Observation   — we run the tool and feed the real result back in.
    ... repeat ... until the model stops asking for tools and just answers.

That loop is the *entire* idea behind "agents". There is no magic: we call the
LLM, and if the reply wants tools we execute them, append the results to the
conversation, and call the LLM again. The model sees the observation on the next
turn and decides whether it now has enough to answer.

`self.trace` records every Thought / Action / Observation so a newcomer (or a
debugger) can read exactly what the agent did and why.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .llm import LLM, Message
from .tools import Registry, registry as default_registry

DEFAULT_SYSTEM = (
    "You are Musaeus, a helpful assistant. You can call tools to look things up "
    "or compute answers. Use a tool when it makes the answer more accurate; "
    "otherwise just reply. When you have enough information, give a final answer."
)


@dataclass
class Step:
    """One readable entry in the trace: a thought, an action, or an observation."""
    kind: str            # "thought" | "action" | "observation"
    content: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.kind}] {self.content}"


class Agent:
    """A ReAct agent: `Agent(llm).run("...")` returns the final text answer."""

    def __init__(
        self,
        llm: LLM,
        registry: Registry = default_registry,
        system: str | None = None,
        max_steps: int = 8,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system = system or DEFAULT_SYSTEM
        self.max_steps = max_steps
        self.trace: list[Step] = []

    def run(self, user_message: str) -> str:
        """Drive the loop until the model gives a text answer or we hit max_steps."""
        # The running conversation. We start with the system prompt + user turn,
        # then grow it with assistant/tool messages as the loop proceeds.
        messages: list[Message] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_message},
        ]
        self.trace = []
        tools = self.registry.schemas()

        for _ in range(self.max_steps):
            reply = self.llm.chat(messages, tools=tools)

            # The model's prose reasoning for this turn — its "Thought".
            if reply.text:
                self.trace.append(Step("thought", reply.text))

            # No tool calls => the model is done reasoning; this text is the answer.
            if not reply.wants_tools:
                return reply.text

            # Otherwise we must (1) record the assistant's tool-call turn, then
            # (2) run each tool and append its result. The assistant turn has to
            # go into `messages` first, because the OpenAI-compatible protocol
            # requires every `tool` result to reference the assistant message that
            # requested it (via tool_call_id). We rebuild that message from the
            # normalized reply so this works regardless of provider.
            messages.append(self._assistant_tool_turn(reply))

            for call in reply.tool_calls:
                self.trace.append(Step(
                    "action", f"{call.name}({json.dumps(call.arguments)})",
                    detail={"id": call.id, "name": call.name, "args": call.arguments},
                ))
                observation = self.registry.run(call.name, call.arguments)
                self.trace.append(Step("observation", observation, detail={"id": call.id}))
                messages.append(self._tool_result(call.id, call.name, observation))

        # Ran out of steps without a plain-text answer. Return the last thought
        # (if any) so the caller still gets the model's best effort.
        last = next((s.content for s in reversed(self.trace) if s.kind == "thought"), "")
        return last or f"(stopped after {self.max_steps} steps without a final answer)"

    # -- message shaping ----------------------------------------------------
    # These build the two message shapes the loop appends, in the agent's ONE
    # internal dialect: the OpenAI-compatible wire format. Providers that speak
    # something else are the llm module's problem — `_to_anthropic_messages` in
    # llm/__init__.py rewrites these into Anthropic content blocks, so this loop
    # never branches on provider.

    @staticmethod
    def _assistant_tool_turn(reply) -> Message:
        """Reconstruct the assistant turn that requested tools (OpenAI shape).

        `arguments` must be a JSON *string* on the wire — the same way the model
        originally emitted it — so we re-serialize the parsed dict.
        """
        return {
            "role": "assistant",
            "content": reply.text or "",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in reply.tool_calls
            ],
        }

    @staticmethod
    def _tool_result(call_id: str, name: str, content: str) -> Message:
        """The observation fed back to the model: a `tool` role message keyed to
        the call it answers by `tool_call_id`."""
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }
