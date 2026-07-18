"""cache.py — two different caches that people constantly conflate.

There are TWO unrelated things called "caching" around an LLM, and mixing them
up leads to bugs. This module is about the second one, but names the first so a
reader can tell them apart:

(1) **Provider-side prompt caching** (e.g. Anthropic `cache_control:
    {"type": "ephemeral"}`).
    You still *call* the model every time. What you save is the model
    re-processing a long, stable *prefix* (a big system prompt, a tool list, a
    document) on each call. The provider keeps the processed prefix warm for a
    few minutes; subsequent calls that reuse that exact prefix pay ~0.1x for it
    and return faster. It cuts cost + latency of repeated large prefixes — it
    does NOT skip the call, and the answer can still differ each time.
    See `with_anthropic_cache()` below for the block shape.

(2) **Response caching** (this module, `ResponseCache`).
    When the *exact same request* (same messages, same model) shows up again,
    don't call the model at all — return the stored reply. This skips the
    network round-trip entirely: zero tokens, zero latency. The tradeoff is it
    only helps on *identical* repeats, and it's only correct when you're happy
    to serve a deterministic, memoised answer (docs Q&A, evals, dev loops).

Rule of thumb: prompt caching makes each call cheaper; response caching removes
the call. They compose — you can do both.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

CACHE_DIR = Path.home() / ".musaeus"
CACHE_FILE = CACHE_DIR / "cache.json"


def cache_key(messages: list[dict[str, Any]], model: str) -> str:
    """A stable hash of (messages, model) — the identity of a request.

    "Stable" is the whole point: the same request must produce the same key
    across processes and Python runs, so we can't use the built-in `hash()`
    (it's salted per-process). We serialise deterministically (`sort_keys` so
    dict order never matters) and SHA-256 the bytes. The model is part of the
    key because the same prompt on a different model is a different request.
    """
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """A tiny request -> reply cache: in-memory dict, mirrored to one JSON file.

    The in-memory layer keeps `get`/`set` cheap within a run; the JSON file
    makes it survive restarts. This is deliberately simple (no eviction, no
    TTL) — it's a teaching cache, not a production one. For a real system you'd
    reach for sqlite or Redis, but the shape of the idea is exactly this.
    """

    def __init__(self, path: Path = CACHE_FILE):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mem: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            # A corrupt cache should never crash the caller — start empty.
            return {}

    def _flush(self) -> None:
        # Create ~/.musaeus on first write, then dump the whole map. Small
        # cache, so rewriting the file wholesale is fine and keeps it atomic-ish.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._mem, ensure_ascii=False, indent=2))
        tmp.replace(self.path)  # atomic swap — never leave a half-written file

    def get(self, key: str) -> Any | None:
        """Return the cached value for `key`, or None on a miss."""
        with self._lock:
            return self._mem.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store `value` under `key` in memory and on disk."""
        with self._lock:
            self._mem[key] = value
            self._flush()


def with_anthropic_cache(system_text: str) -> dict[str, Any]:
    """Show the shape of a provider-side *prompt cache* breakpoint (concept 1).

    This is NOT the response cache above — it returns the `system` block you'd
    pass to Anthropic's /v1/messages so the model keeps your long, stable system
    prompt warm. The `cache_control` marker says "cache everything up to here";
    put it on the last stable block, and keep volatile content (the user's
    varying question) *after* it so the cached prefix stays byte-identical.
    """
    return {
        "system": [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},  # 5-min warm prefix
            }
        ]
    }
