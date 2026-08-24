"""tools/ — the tool registry: how an LLM gets *hands*.

A language model can only emit text. "Function calling" is the trick that turns
that text into actions: you hand the model a list of **tool schemas** (name +
description + a JSON Schema for the arguments), and instead of answering in prose
it can answer with a structured request — "call `calculator` with
{"expression": "2+2"}". Your code runs the real Python function and feeds the
result back. The model never runs anything itself; it only *asks*, and this
registry is what actually does the work.

This module gives you three pieces:
  - `@tool`   — decorate a plain Python function to register it as a tool. The
                schema is inferred from the function's type hints + docstring, so
                there is exactly one source of truth: the function itself.
  - `Tool`    — the dataclass holding a registered function and its schema.
  - `Registry`— `.schemas()` (what you pass to `LLM.chat(tools=...)`) and
                `.run(name, args)` (what the agent calls to execute a tool).

Plus a module-level default `registry` and three working example tools.
"""
from __future__ import annotations

import ast
import inspect
import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, get_origin, get_type_hints

# --- type hint -> JSON Schema -----------------------------------------------
# The model needs JSON Schema, not Python types. This is the whole mapping; if a
# hint isn't here we fall back to "string", which every provider accepts.
_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_for(func: Callable) -> dict[str, Any]:
    """Infer a JSON-Schema `parameters` object from a function's signature.

    Each parameter becomes a property (typed via its hint); parameters without a
    default become `required`. The function's docstring becomes the tool's
    description — so documenting the function *is* documenting the tool.
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        hint = hints.get(name, str)
        # `list[str]` is not `list` — generics hide their base type behind
        # get_origin(). Anything we still can't name degrades to "string",
        # which every provider accepts.
        json_type = _JSON_TYPES.get(hint) or _JSON_TYPES.get(get_origin(hint), "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


@dataclass
class Tool:
    """A single callable exposed to the model, paired with its schema."""
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        # The exact "tool schema" shape the LLM interface expects (see llm/).
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class Registry:
    """A name -> Tool table. The agent asks it for schemas and to run tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        """The list you pass to `LLM.chat(tools=...)` — one dict per tool."""
        return [t.schema() for t in self._tools.values()]

    def run(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name and return its result **as a string**.

        Results are always stringified because that's what goes back into the
        conversation as the `tool` message content. We catch errors here instead
        of raising: a failed tool should tell the model what went wrong so it can
        recover, not crash the whole agent loop.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"error: no such tool {name!r} (have: {', '.join(self._tools) or 'none'})"
        try:
            result = tool.func(**args)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the model
            return f"error: {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else str(result)


# The default registry every module shares. Import it, or make your own Registry.
registry = Registry()


def tool(func: Callable[..., Any] | None = None, *, reg: Registry = registry) -> Callable:
    """Register a function as a tool (on `registry` by default).

    Usage — bare or with a target registry::

        @tool
        def now() -> str: ...

        @tool(reg=my_registry)
        def calculator(expression: str) -> str: ...

    The function keeps working as a normal Python callable; the decorator just
    also files it (name, docstring, inferred schema) into the registry.
    """
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        reg.add(Tool(
            name=fn.__name__,
            description=(inspect.getdoc(fn) or "").strip(),
            func=fn,
            parameters=_schema_for(fn),
        ))
        return fn

    return wrap(func) if func is not None else wrap


# --- example tools (real, working) ------------------------------------------
# A tiny allow-list of AST nodes = a calculator that can't `import os` or call
# arbitrary functions. `eval()` on user text would be a remote-code-execution
# hole; this walks the parse tree and only permits arithmetic.
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. "3 * (4 + 5) ** 2".

    Supports + - * / // % ** and parentheses. No variables, names, or functions.
    """
    return str(_safe_eval(ast.parse(expression, mode="eval").body))


@tool
def now() -> str:
    """Return the current date and time in UTC (ISO-8601)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents (capped at 20,000 characters)."""
    cap = 20_000
    data = Path(path).read_text(encoding="utf-8", errors="replace")
    return data if len(data) <= cap else data[:cap] + f"\n...[truncated at {cap} characters]"
