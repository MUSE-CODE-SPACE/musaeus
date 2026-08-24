"""embed.py — turn text into vectors, online if we can, offline if we must.

An embedding maps a passage to a point in high-dimensional space where "close"
means "similar in meaning". That's the whole engine behind retrieval: embed the
query, find the nearest passages.

Two paths, and the trade-off between them:

  1. Provider embeddings (preferred). We POST to the OpenAI-compatible
     `/embeddings` endpoint — the same shape OpenAI serves and that Ollama serves
     locally (e.g. `nomic-embed-text`). These are *learned* embeddings: they know
     that "car" and "automobile" are neighbours. Quality is far higher, but it
     needs a running model/service and can cost money or latency.

  2. Local hashing fallback (labeled, deterministic, offline). If no endpoint
     answers, we build a vector from hashed word features. This is HONEST but
     DUMB: it captures lexical overlap only — "car" and "automobile" are strangers
     to it. It exists so the whole RAG pipeline runs, is testable, and never
     crashes with zero network. Never ship it as your real retriever; it's the
     "batteries included so the lights turn on" option.

numpy is a hard dependency here (the `rag` extra) — vector math wants it.
"""
from __future__ import annotations

import hashlib

import httpx
import numpy as np

from ..config import Settings

# Dimensionality of the local fallback vectors. Fixed so a store built offline
# stays self-consistent; unrelated to any provider model's real dimension.
_HASH_DIM = 512


def embed(texts: list[str], settings: Settings) -> list[list[float]]:
    """Embed `texts`, trying the provider endpoint first, else hashing locally.

    Returns one vector (list[float]) per input string, in order.
    """
    if not texts:
        return []
    vecs = _embed_via_provider(texts, settings)
    if vecs is not None:
        return vecs
    return _embed_local_hashing(texts)


def _embed_via_provider(texts: list[str], s: Settings) -> list[list[float]] | None:
    """Call the OpenAI-compatible /embeddings endpoint. Return None if unavailable.

    We mirror `llm/__init__.py`: `local` points at the Ollama gateway (any bearer
    key works), `openai` at api.openai.com. We swallow connection/HTTP errors and
    signal "unavailable" with None so the caller can fall back — offline is a
    supported mode, not an exception to shout about.
    """
    if s.provider == "anthropic":
        return None  # Anthropic serves no OpenAI-style embeddings endpoint.
    if s.provider == "local":
        base, key, model = s.local_base_url, "ollama", "nomic-embed-text"
    elif s.provider == "google":
        # Gemini's OpenAI-compatibility layer serves /embeddings too.
        base, key, model = (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            s.google_api_key or "",
            "gemini-embedding-001",
        )
    else:
        base, key, model = "https://api.openai.com/v1", s.openai_api_key or "", "text-embedding-3-small"

    try:
        r = httpx.post(
            f"{base}/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "input": texts},
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()["data"]
        # The endpoint may return rows out of order; `index` restores it.
        rows = sorted(data, key=lambda d: d["index"])
        return [row["embedding"] for row in rows]
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def _embed_local_hashing(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-hashed-words embedding — FALLBACK ONLY, offline.

    Each lowercased word is hashed into one of `_HASH_DIM` buckets (the "hashing
    trick": no vocabulary to store, fixed width). We L2-normalize so cosine
    similarity later reduces to a dot product, and two texts sharing many words
    score high. It knows nothing of meaning — only spelling.
    """
    out: list[list[float]] = []
    for text in texts:
        v = np.zeros(_HASH_DIM, dtype=np.float32)
        for word in text.lower().split():
            h = hashlib.sha1(word.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % _HASH_DIM
            sign = 1.0 if h[4] & 1 else -1.0  # signed hashing curbs collision bias
            v[bucket] += sign
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v /= norm
        out.append(v.tolist())
    return out
