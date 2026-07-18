"""store.py — a tiny in-memory vector store with exact cosine search.

Once passages are vectors, "retrieval" is just: which stored vectors point most
nearly the same direction as the query vector? That direction-similarity is
*cosine similarity* — the dot product of two L2-normalized vectors, in [-1, 1],
where 1 means "same direction / most similar".

Exact vs approximate (ANN):
  We do EXACT search — compare the query against every stored vector. It's O(n·d)
  per query, dead simple, and always returns the true top-k. Perfect for a
  teaching store and up to ~10^5 vectors.
  At real scale (millions of vectors) that linear scan gets slow, so production
  systems use Approximate Nearest Neighbour indexes — HNSW (a navigable
  small-world graph) or IVF (inverted file / coarse clustering), as in FAISS,
  hnswlib, or a vector DB. They trade a sliver of recall for a huge speed-up. We
  intentionally skip them: the concept here is the cosine ranking, not the index.

numpy is a hard dependency (the `rag` extra): we keep all vectors as one 2-D
matrix so a search is a single matrix–vector product.
"""
from __future__ import annotations

import numpy as np


class VectorStore:
    """Holds (id, vector, text) rows and answers nearest-neighbour queries."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.texts: list[str] = []
        # Vectors live in one (n, d) float32 matrix, L2-normalized on add, so
        # search is `matrix @ query` — cosine similarity as a single dot product.
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vectors: list[list[float]], texts: list[str]) -> None:
        """Append rows. `ids`, `vectors`, and `texts` must line up 1:1."""
        if not (len(ids) == len(vectors) == len(texts)):
            raise ValueError("ids, vectors, and texts must be the same length")
        if not ids:
            return
        block = _normalize(np.asarray(vectors, dtype=np.float32))
        self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])
        self.ids.extend(ids)
        self.texts.extend(texts)

    def search(self, query_vec: list[float], k: int = 4) -> list[tuple[str, float]]:
        """Return the k most cosine-similar (text, score) pairs, best first."""
        if self._matrix is None or not self.ids:
            return []
        q = _normalize(np.asarray([query_vec], dtype=np.float32))[0]
        # Normalized rows · normalized query == cosine similarity for every row.
        scores = self._matrix @ q
        k = min(k, len(self.ids))
        # argpartition grabs the top-k cheaply; then sort just those k by score.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.texts[i], float(scores[i])) for i in top]


def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize each row so later dot products equal cosine similarity.

    Guard the zero-vector case (norm 0) to avoid divide-by-zero: a zero row stays
    zero and simply scores 0 against everything.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms
