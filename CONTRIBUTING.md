# Contributing to Musaeus

Musaeus is a teaching codebase first: every module is a self-contained lesson, and a
change that makes the code faster but harder to read is usually a net loss here. Match
the existing style — read `src/musaeus/config.py` and `src/musaeus/llm/__init__.py`
before writing anything, they set the tone.

## House style

- **Dependency-light.** Prefer the stdlib and what's already in `pyproject.toml`
  (httpx, pydantic, typer, rich; numpy only under the `rag` extra; fastapi/uvicorn under
  `server`). A new dependency needs a very good reason.
- **One module = one concept.** A newcomer reading a file should come away able to say
  "ah, THIS is what a guardrail / RAG / ReAct loop is." Comment the *why*, not the obvious.
- **Real, working code.** No `pass`-only stubs. If something needs a running model or
  service, guard it so the module still imports and its pure logic stays testable offline.
- **Python 3.10+**, type hints, `from __future__ import annotations`.
- Keep files focused — roughly 60–200 lines. No frameworks, no clever abstractions.

## Public interfaces

Modules compose through these names; changing a signature means updating every caller
and the README's concept map.

```python
from musaeus.config import Settings, load_settings
from musaeus.llm import build_llm, LLM, Reply, ToolCall
# LLM.chat(messages, tools=None, temperature=0.7, max_tokens=4096) -> Reply
from musaeus.tools import tool, Tool, Registry, registry
from musaeus.agent import Agent            # Agent(llm).run("...") -> str
from musaeus.schema import extract         # extract(llm, prompt, Model) -> Model
from musaeus.guardrails import scan_input, scan_output, GuardResult
from musaeus.rag import Rag                # .ingest(texts), .retrieve(query, k)
from musaeus.eval import judge, run_eval
from musaeus.mcp import MCPClient
```

A "tool schema" everywhere in Musaeus is `{"name", "description", "parameters": <JSON Schema>}`.
The agent's internal message dialect is the OpenAI-compatible wire shape; translations to
other providers live in `llm/` and nowhere else.

## Checks

```bash
uv sync --all-extras          # or: pip install -e ".[dev,rag,server]"
pytest                        # offline — no keys, no network
ruff check src tests
```

Bug reports with a failing test are the fastest path to a merge.
