"""Offline tests for the provider translation layer — no network, no keys.

The agent speaks one internal dialect (the OpenAI-compatible message shape);
`llm/` owns every translation out of it. These tests pin that translation down,
because it's exactly the part a mocked end-to-end test can't see failing.
"""
from __future__ import annotations

import json

from musaeus.llm import _supports_sampling, _to_anthropic_messages
from musaeus.tools import _schema_for


def test_anthropic_translation_moves_system_to_top_level():
    system, convo = _to_anthropic_messages([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ])
    assert system == "be brief"
    assert convo == [{"role": "user", "content": "hi"}]


def test_anthropic_translation_rewrites_tool_turns():
    # The exact history an Agent builds after one tool call, OpenAI-shaped.
    messages = [
        {"role": "user", "content": "what is 6*7?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "calculator", "arguments": json.dumps({"expression": "6*7"})},
            }],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "calculator", "content": "42"},
    ]
    _, convo = _to_anthropic_messages(messages)

    assistant = convo[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [
        {"type": "tool_use", "id": "c1", "name": "calculator", "input": {"expression": "6*7"}}
    ]
    result = convo[2]
    assert result["role"] == "user"  # tool results become user-side content blocks
    assert result["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": "42"}
    ]


def test_anthropic_translation_merges_parallel_tool_results():
    # Two results answering ONE assistant turn must ride in a single user message.
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant", "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "now", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "now", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "name": "now", "content": "t1"},
        {"role": "tool", "tool_call_id": "b", "name": "now", "content": "t2"},
    ]
    _, convo = _to_anthropic_messages(messages)
    assert len(convo) == 3
    assert [b["tool_use_id"] for b in convo[2]["content"]] == ["a", "b"]


def test_sampling_lock_covers_frontier_models_only():
    assert _supports_sampling("gemma3")
    assert _supports_sampling("claude-sonnet-4-6")
    assert not _supports_sampling("claude-opus-5")
    assert not _supports_sampling("gpt-5-mini")


def test_tool_schema_understands_generic_hints():
    def sample(names: list[str], limit: int = 3) -> str:
        """Doc."""
        return ""

    params = _schema_for(sample)
    assert params["properties"]["names"]["type"] == "array"   # list[str] -> array
    assert params["properties"]["limit"]["type"] == "integer"
    assert params["required"] == ["names"]
