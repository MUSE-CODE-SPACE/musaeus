"""rag/ — retrieval-augmented generation: the agent's memory of your documents.

An LLM only knows what's in its weights and what's in the prompt. RAG is how you
give it the *second* part on demand: instead of fine-tuning the model on your
docs, you store your docs as vectors and, at question time, paste the few most
relevant passages into the prompt. The model reasons over facts it was never
trained on — and you can cite exactly where each fact came from.

The pattern is three steps, and this package is those three steps:

    retrieve  →  augment  →  generate

  - retrieve: embed the user's question, find the nearest stored passages
              (`embed.py` + `store.py`).
  - augment:  splice those passages into the prompt as grounding context. That
              step lives in the *caller* (the agent/prompt), not here — this
              module's job is to hand back the right passages. `retrieve()` is the
              seam.
  - generate: the LLM answers using that context. Same `LLM.chat` as everywhere
              else; RAG changes the *prompt*, not the model.

Ingestion mirrors it: chunk the documents (`chunk.py`), embed the chunks, store
them. Do that once up front; retrieve on every query.
"""
from __future__ import annotations

from ..config import Settings, load_settings
from .chunk import chunk_text
from .embed import embed
from .store import VectorStore


class Rag:
    """Document memory: ingest texts once, retrieve relevant passages per query."""

    def __init__(self, settings: Settings | None = None) -> None:
        # Settings decides where embeddings come from (provider endpoint vs the
        # offline hashing fallback) — see embed.py. Default to load_settings() so
        # a Rag() with no args still works.
        self.settings = settings or load_settings()
        self.store = VectorStore()
        self._next_id = 0  # monotonic chunk ids; the store just needs them unique.

    def ingest(self, texts: list[str]) -> int:
        """chunk → embed → store. Returns how many chunks were added.

        Each input document is chunked independently (so chunks never straddle a
        document boundary), then all new chunks are embedded in one batch and
        stored. Call this as often as you like; chunks accumulate.
        """
        chunks: list[str] = []
        for text in texts:
            chunks.extend(chunk_text(text))
        if not chunks:
            return 0
        vectors = embed(chunks, self.settings)
        ids = [str(i) for i in range(self._next_id, self._next_id + len(chunks))]
        self._next_id += len(chunks)
        self.store.add(ids, vectors, chunks)
        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        """Return the k passages most relevant to `query`, best first.

        This is the 'retrieve' in retrieve→augment→generate. The caller pastes
        these into the prompt as context; we deliberately return plain strings
        (not scores) so that wiring stays trivial.
        """
        [query_vec] = embed([query], self.settings)
        return [text for text, _score in self.store.search(query_vec, k)]


__all__ = ["Rag", "chunk_text", "embed", "VectorStore"]
