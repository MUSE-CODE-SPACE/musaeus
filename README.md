# Musaeus

**An open, local-first LLM agent you actually own — and a working map of how modern LLM systems are built.**

Musaeus (µουσαῖος, *"of the Muses"*) is a from-scratch, permissively-licensed AI assistant in the spirit of open models like Nous Research's Hermes: capable, steerable, and yours. It runs on a local open model by default and can switch to a cloud provider with one line.

But Musaeus is also a **teaching codebase**. Every core capability lives in its own clearly-named module, so the architecture reads like a syllabus. Open `guardrails.py` and you are looking at exactly what a guardrail is. Open `rag/` and you are looking at retrieval-augmented generation, one honest step at a time. Nothing is hidden behind a framework.

> Built live in the [Musaeus build-first course](#) — we start from `git init`, and pull in each concept the moment the build needs it.

---

## Why this exists

Most "build an AI agent" tutorials either (a) `pip install` a giant framework and wave their hands, or (b) drown you in theory before you write a line. Musaeus takes the third path: **build a real thing, and name every concept as it appears.** By the end you have a working assistant *and* you can point at each idea — tool use, structured output, RAG, guardrails, evaluation — and say what it is and why it's there.

## The concept map — one module, one idea

Each module is a self-contained lesson. The chapter references point to the full LLM course for the deep dive.

| Module | The concept, in one line | Deep dive |
|---|---|---|
| `llm/` | Talk to a model — local (Ollama) or cloud, one interface | SDK comparison |
| `sampling.py` | The knobs that shape every output (temperature, top-p, stop) | Foundations |
| `tools/` | **Function calling** — give the model hands it can actually use | Tool use |
| `schema.py` | **Structured output** — outputs your code can trust (Pydantic) | Structured output |
| `rag/` | **RAG** — chunk, embed, retrieve; give the agent memory of *your* docs | RAG & Vector DBs |
| `agent.py` | The **ReAct loop** — reason, act, observe, repeat | Agent engineering |
| `guardrails.py` | **Guardrails** — validate input/output, block prompt injection | Production security |
| `cache.py` | **Prompt caching** — make the same app faster *and* cheaper | Context & caching |
| `eval/` | **LLM-as-judge evaluation** — prove it still works after you change it | Production eval |
| `router.py` | **Cost-aware routing** — easy → cheap model, hard → strong model | Production cost |
| `mcp/` | **MCP** — discover and call external tools over a standard protocol | MCP deep dive |
| `server.py` | Serve it — **FastAPI** with token **streaming** | FastAPI |
| `cli.py` | A real terminal assistant you can run today | — |

## Quickstart

```bash
# 1. clone + install (uv recommended)
git clone https://github.com/MUSE-CODE-SPACE/musaeus
cd musaeus && uv sync           # or: pip install -e .

# 2. run fully local (no API key) — needs Ollama + an open model
ollama pull gemma3              # or llama3.1, qwen2.5, hermes3
musaeus chat                    # talk to your local assistant

# 3. or point it at a cloud provider
cp .env.example .env            # add ANTHROPIC_API_KEY / OPENAI_API_KEY
musaeus chat --provider anthropic
```

## Design principles

1. **Local-first, cloud-optional.** Your assistant should run on hardware you own.
2. **No hidden magic.** Every capability is readable, single-responsibility code — no framework lock-in.
3. **Name the concept.** Modules are named after the idea they implement, not after clever abstractions.
4. **Provider-agnostic.** Anthropic, OpenAI, Google, and OpenAI-compatible local gateways behind one interface.

## Status

Musaeus is built in the open as an educational capstone. It is a clean-room reimplementation of *capabilities* found in open assistants — it ships no third-party weights, code, or branding. Model weights (Gemma, Llama, Hermes, etc.) are downloaded by you, from their own sources, under their own licenses.

## License

Apache-2.0 — see [LICENSE](LICENSE). Do what you like; attribution appreciated.
