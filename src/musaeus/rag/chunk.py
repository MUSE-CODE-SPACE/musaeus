"""chunk.py — cut a document into overlapping windows for retrieval.

Why chunk at all? An embedding model turns a *span* of text into one vector. Feed
it a whole 40-page document and you get a single blurry vector that means
"everything, and therefore nothing" — retrieval can't point at the paragraph that
actually answers the question. So we slice the text into small passages and embed
each one; retrieval then returns the few passages that matter.

Two knobs, and the tension between them:
  - `size`   — how many characters per chunk. Too big and a chunk mixes several
               ideas (imprecise matches, and you waste the LLM's context on
               irrelevant text). Too small and a chunk loses the context that
               makes it meaningful ("it raised $2M" — *what* raised it?).
  - `overlap`— how many characters each chunk shares with the previous one. A hard
               cut every `size` characters will slice a sentence — and the fact
               you need — straight down the middle, so it lands in *neither*
               chunk's vector cleanly. Overlap is a sliding window: the boundary
               sentence appears whole in one of the two neighbours.

Defaults (800 / 120) are a decent middle for prose. Tune by content: code and
tables often want larger, non-overlapping chunks aligned to natural boundaries.
We measure in characters (not tokens) to stay dependency-free; ~4 chars ≈ 1 token,
so 800 chars is roughly a 200-token passage.
"""
from __future__ import annotations


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split `text` into overlapping windows of ~`size` chars, sharing `overlap`.

    The window advances by `size - overlap` each step, so consecutive chunks share
    their boundary region. Returns non-empty, stripped chunks in document order.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        # overlap must leave forward progress, else the window never advances.
        raise ValueError("overlap must be in [0, size)")

    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break  # last window already reached the end; don't emit tail dupes.
    return chunks
