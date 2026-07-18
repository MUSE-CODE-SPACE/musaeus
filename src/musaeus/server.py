"""server.py — Musaeus as an HTTP API, streaming answers over SSE.

Two endpoints:
  - GET  /health  -> a liveness probe (no model needed).
  - POST /chat    -> {"message": str, "provider"?: str}, answer streamed as
                     Server-Sent Events (text/event-stream).

Why stream? An LLM produces its answer left-to-right over seconds. If we buffered
the whole reply and sent it in one shot, the user would stare at a spinner the
entire time. Streaming turns that dead air into visible progress — the first words
appear almost immediately — which is the single biggest perceived-latency win in
LLM UX. SSE is the least-effort transport for this: a plain HTTP response whose
body is a sequence of `data: ...\n\n` frames the browser's EventSource consumes.

FastAPI/uvicorn live under the optional `server` extra. We import them LAZILY so
`import musaeus.server` (e.g. for tests, or on a box without the extra) never
crashes — the heavy imports only happen when you actually build or run the app.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .agent import Agent
from .config import load_settings
from .llm import build_llm

# Frame that tells an EventSource client the stream is complete.
_DONE = "data: [DONE]\n\n"


def _sse(payload: dict[str, Any]) -> str:
    """Encode one dict as a single SSE `data:` frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_agent(provider: str | None) -> Agent:
    """load_settings -> build_llm -> Agent, the same wiring the CLI uses."""
    settings = load_settings(provider=provider)
    return Agent(build_llm(settings))


async def _stream_answer(message: str, provider: str | None):
    """Async generator of SSE frames for one /chat request.

    `Agent.run` is synchronous and returns the *whole* answer (its ReAct loop may
    call tools between model turns, so there's no token stream to forward). We run
    it off the event loop with `asyncio.to_thread` so one slow request can't block
    every other connection, then chunk the finished text into small frames. The
    client still renders progressively — the honest ceiling given a blocking Agent.
    A future token-streaming `Agent` would just yield straight into `_sse` here.
    """
    agent = _build_agent(provider)
    try:
        answer = await asyncio.to_thread(agent.run, message)
    except Exception as exc:  # surface errors as a frame instead of a dead socket
        yield _sse({"error": type(exc).__name__, "detail": str(exc)})
        yield _DONE
        return

    for i in range(0, len(answer), 24):
        yield _sse({"delta": answer[i : i + 24]})
        await asyncio.sleep(0)  # cooperatively flush each frame
    yield _DONE


def create_app():
    """Build the FastAPI app. Imports fastapi lazily (needs the `server` extra)."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            "FastAPI is not installed. Install the server extra:\n"
            "    pip install 'musaeus[server]'"
        ) from exc

    app = FastAPI(title="Musaeus", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        message = (body or {}).get("message")
        if not isinstance(message, str) or not message.strip():
            return JSONResponse({"error": "field 'message' (str) is required"}, status_code=400)
        provider = (body or {}).get("provider")
        # StreamingResponse drives our async generator; the media type is what
        # makes browsers treat the body as an SSE stream. Disable proxy buffering
        # so frames reach the client the moment we yield them.
        return StreamingResponse(
            _stream_answer(message, provider),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


# PEP 562 module-level lazy attribute: `musaeus.server.app` builds the FastAPI app
# on first access (and caches it), so the plain `import musaeus.server` above stays
# cheap and never needs fastapi installed. `uvicorn musaeus.server:app` still works.
_app_cache: Any = None


def __getattr__(name: str) -> Any:
    global _app_cache
    if name == "app":
        if _app_cache is None:
            _app_cache = create_app()
        return _app_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """Console/`python -m` entry point: serve the app with uvicorn."""
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise SystemExit(
            "uvicorn is not installed. Install the server extra:\n"
            "    pip install 'musaeus[server]'"
        ) from exc
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
