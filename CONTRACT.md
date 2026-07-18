# Musaeus module contract (READ FIRST)

You are implementing part of **Musaeus**, an open local-first LLM agent that is ALSO a
teaching codebase. Match the existing style exactly — read these two files first:
- `src/musaeus/config.py`  (Settings, load_settings)
- `src/musaeus/llm/__init__.py`  (the LLM interface you will build on)

## House style (non-negotiable)
- **Dependency-light.** Prefer stdlib + what's already in `pyproject.toml` (httpx, pydantic,
  typer, rich; numpy only under the `rag` extra; fastapi/uvicorn under `server`). Do not add new deps.
- **One module = one concept, legible.** A newcomer reading the file should come away able to say
  "ah, THIS is what a guardrail / RAG / ReAct loop is." Comment the *why*, not the obvious.
- **Real, working code.** No `pass`-only stubs. If something needs a running model/service, guard it
  so the module still imports and its pure logic is testable offline.
- **Python 3.10+, type hints, `from __future__ import annotations`.**
- Keep each file focused and readable (roughly 60–160 lines). No frameworks.

## Interfaces you can rely on (already built)
```python
from musaeus.config import Settings, load_settings
from musaeus.llm import build_llm, LLM, Reply, ToolCall
# LLM.chat(messages: list[dict], tools: list[dict]|None=None,
#          temperature=0.7, max_tokens=1024) -> Reply
# Reply.text: str ; Reply.tool_calls: list[ToolCall] ; Reply.wants_tools: bool
# ToolCall.id/name/arguments(dict)
# A "tool schema" dict is: {"name": str, "description": str, "parameters": <JSON Schema dict>}
```

## Interfaces you must EXPOSE (so other modules compose)
Implement these public names exactly; other agents depend on them.

- `tools/__init__.py` — a tool registry:
  - `@tool` decorator: registers a Python function as a tool (name from fn, description from docstring,
    JSON-Schema parameters inferred from type hints).
  - `class Registry`: `.schemas() -> list[dict]` (tool-schema dicts), `.run(name, args: dict) -> str`.
  - a module-level `registry` (default Registry) + a few example tools (e.g. `calculator`, `now`, `read_file`).
- `agent.py` — the ReAct loop:
  - `class Agent`: `Agent(llm: LLM, registry=registry, system: str|None=None, max_steps: int=8)`
  - `.run(user_message: str) -> str` — loops: chat → if `reply.wants_tools`, run each tool and append
    `tool` results to the message list → repeat until a text answer or max_steps. Return final text.
  - Keep a readable transcript on `self.trace` (list of steps) for teaching/debugging.
- `schema.py` — structured output:
  - `extract(llm: LLM, prompt: str, model: type[BaseModel], max_retries: int = 2) -> BaseModel`
    — force the model to return JSON matching `model`, validate with Pydantic, retry on failure.
- `guardrails.py`:
  - `@dataclass GuardResult: ok: bool; reasons: list[str]`
  - `scan_input(text: str) -> GuardResult` (prompt-injection / obvious-attack heuristics)
  - `scan_output(text: str, deny: list[str]|None=None) -> GuardResult`
- `sampling.py`:
  - `PRESETS: dict[str, dict]` (e.g. "precise", "balanced", "creative" → temperature/top_p) + a short
    docstring explaining each knob.
- `cache.py`:
  - `class ResponseCache`: `.get(key)`, `.set(key, value)` (a simple on-disk/in-mem prompt→reply cache)
  - `cache_key(messages, model) -> str`; plus a docstring explaining provider prompt-caching (Anthropic
    `cache_control: {"type":"ephemeral"}`) vs this local response cache.
- `router.py`:
  - `route(message: str) -> str` returning a provider name ("local"/"anthropic"/"openai") by a cheap
    difficulty heuristic; docstring explains cost-aware routing.
- `rag/__init__.py`:
  - `class Rag`: `.ingest(texts: list[str])`, `.retrieve(query: str, k: int = 4) -> list[str]`.
    Split into `rag/chunk.py`, `rag/embed.py`, `rag/store.py` (in-memory numpy cosine store).
    Embeddings: call the provider's embeddings endpoint if available, else a clearly-labeled local
    hashing fallback so it runs offline.
- `eval/__init__.py`:
  - `judge(llm: LLM, question: str, answer: str, rubric: str) -> dict` (score 1-5 + rationale via
    LLM-as-judge) and `run_eval(cases: list[dict], answer_fn) -> dict` (aggregate).
- `mcp/__init__.py`:
  - a minimal MCP client over stdio (JSON-RPC 2.0): `class MCPClient` with `.initialize()`,
    `.list_tools() -> list[dict]`, `.call_tool(name, args) -> str`. Faithful to the real protocol
    shapes; degrade gracefully if no server is connected.
- `server.py` — FastAPI: `app` with `POST /chat` that runs an `Agent` and streams tokens (SSE).
  Import fastapi lazily so the package still imports without the `server` extra.
- `cli.py` — a `typer` app named `app`: `musaeus chat [--provider] [--model]` runs an interactive REPL
  built on `Agent`, using `rich` for output.

## Deliverable
Write your assigned file(s) under `/Users/muse/Projects/Musaeus/src/musaeus/`. Then reply with one line:
`<module> done — <n> files`.
