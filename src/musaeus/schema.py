"""schema.py — structured output: making a model return data you can *trust*.

An LLM speaks free text. But an application usually wants a *record* — a person
with a name and an age, an order with typed line items — that the rest of the
program can use without string-parsing prose. That gap is the "structured output"
problem, and it is harder than it looks:

  1. Models drift. Ask for JSON and you may still get a friendly preamble
     ("Sure! Here's the JSON:"), a trailing note, or ```json fences.
  2. JSON that parses is not JSON that's *valid*. `{"age": "old"}` is legal JSON
     but a lie if `age` must be an int.
  3. Failures are random, not systematic. The same prompt succeeds 9 times and
     mangles the 10th.

The reliable recipe is not "prompt harder" — it is a control loop:

    ask -> parse -> VALIDATE against a schema -> on failure, feed the error back
    and retry.

Pydantic gives us the schema (a `BaseModel`) and does the validating; the retry
turns a probabilistic generator into something that behaves like a typed function.
Two forces cooperate: we *push* the model toward the shape (a JSON-Schema tool it
can call, or an instruction with the schema inlined) and we *check* what comes
back. Neither alone is enough — the check is what makes it dependable.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:  # avoid importing the LLM (and httpx) just to type-hint it
    from .llm import LLM


def _tool_from_model(model: type[BaseModel]) -> dict[str, Any]:
    """Turn a Pydantic model into a tool schema the LLM layer understands.

    Our LLM interface takes tools shaped like
    {"name", "description", "parameters": <JSON Schema>}. A Pydantic model already
    *is* a JSON Schema via `model_json_schema()`, so the cleanest way to force
    structure is to offer exactly one tool and ask the model to call it: providers
    constrain tool arguments to the schema far more reliably than they honor a
    "please output JSON" instruction in free text.
    """
    return {
        "name": "emit_" + _snake(model.__name__),
        "description": f"Return the extracted {model.__name__} as structured fields.",
        "parameters": model.model_json_schema(),
    }


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _extract_json(text: str) -> str:
    """Best-effort recovery of a JSON object from a chatty text reply.

    Used only on the fallback path (a model that ignored the tool and just talked).
    We strip ```json fences and, failing that, grab the outermost {...} span. This
    is deliberately forgiving — validation downstream is the real gatekeeper.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    braces = re.search(r"\{.*\}", text, re.DOTALL)
    return braces.group(0) if braces else text


def extract(
    llm: LLM,
    prompt: str,
    model: type[BaseModel],
    max_retries: int = 2,
) -> BaseModel:
    """Extract a validated `model` instance from `prompt` using `llm`.

    Strategy: offer the model a single tool whose parameters *are* the target
    schema and ask it to call that tool. If it calls the tool, its arguments are
    already JSON-shaped and we validate them. If it just replies with text
    (common with smaller local models), we scrape JSON out of the reply. Either
    way Pydantic is the judge — and on a `ValidationError` we loop, appending the
    exact error text so the model can *correct itself* rather than guess again.

    Raises the last `ValidationError` (or a `ValueError` if no JSON ever appeared)
    once retries are exhausted, so callers fail loudly instead of on bad data.
    """
    tool = _tool_from_model(model)
    schema_text = json.dumps(model.model_json_schema(), indent=2)

    system = (
        "You extract structured data. Call the provided tool with fields that "
        "satisfy this JSON Schema exactly. Do not invent fields; use null only "
        "where the schema allows it.\n\nSchema:\n" + schema_text
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    # attempts = the first try + one per allowed retry.
    for attempt in range(max_retries + 1):
        reply = llm.chat(messages, tools=[tool], temperature=0.0)

        # Pull the candidate object from whichever path the model took.
        if reply.wants_tools:
            candidate = reply.tool_calls[0].arguments
        else:
            try:
                candidate = json.loads(_extract_json(reply.text))
            except json.JSONDecodeError as e:
                last_error = ValueError(f"model returned no parseable JSON: {e}")
                _push_retry(messages, reply.text, str(last_error))
                continue

        try:
            return model.model_validate(candidate)
        except ValidationError as e:
            # The key teaching move: hand the *validator's own words* back to the
            # model. "field 'age': input should be a valid integer" is a far more
            # useful correction signal than repeating the original instruction.
            last_error = e
            _push_retry(messages, json.dumps(candidate), e.errors())

    # Exhausted retries — surface the real reason, not a silent bad object.
    raise last_error if last_error else ValueError("extraction failed with no error")


def _push_retry(messages: list[dict[str, Any]], prior: str, error: Any) -> None:
    """Append the failed answer and the validation error, asking for a fix.

    We record the model's own prior output as an assistant turn so it has context
    for what to change, then a user turn with the machine-readable error.
    """
    messages.append({"role": "assistant", "content": prior})
    messages.append(
        {
            "role": "user",
            "content": (
                "That did not validate. Fix these errors and call the tool again "
                "with corrected values:\n" + json.dumps(error, default=str)
            ),
        }
    )
