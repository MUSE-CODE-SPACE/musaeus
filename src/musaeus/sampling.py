"""sampling.py — the knobs that decide *how* a model picks its next word.

At each step a language model produces a probability for every possible next
token. It does NOT just take the most likely one — it *samples*. The sampling
parameters shape that draw, trading determinism for diversity. Understanding them
is the difference between "the model is random" and "I chose how random it is."

The knobs, in plain language:

  temperature  — flattens or sharpens the probability distribution before we draw.
                 0.0 ≈ always take the top token (near-deterministic, repeatable).
                 Higher (→1.0 and beyond) makes unlikely tokens more competitive,
                 so output gets more varied and surprising — and more error-prone.
                 Reach for LOW when there's a right answer (extraction, code,
                 classification); HIGH when you want range (brainstorming, prose).

  top_p        — "nucleus" sampling. Sort tokens by probability, keep the smallest
                 set whose probabilities sum to p, sample only from those. top_p=0.9
                 drops the long tail of nonsense while keeping real alternatives.
                 It's an alternative *cutoff* to temperature; tuning both at once
                 fights itself, so most presets move one and pin the other.

  top_k        — a simpler cutoff: keep only the k most likely tokens, discard the
                 rest. top_k=1 is greedy decoding. Coarser than top_p (fixed count
                 vs. fixed mass); not every provider exposes it, so we keep it
                 optional and out of the default presets.

  max_tokens   — a hard cap on how many tokens to GENERATE (not the context size).
                 Not a quality knob — a budget and a safety valve. Too low truncates
                 mid-sentence (and mid-JSON); it also bounds cost and latency.

  stop         — string(s) that halt generation the moment they appear, before the
                 model rambles past the part you wanted. Useful for delimiting turns
                 or ending a list. The stop string itself is not included in output.

Presets below are starting points, not laws — measure on your own task. They set
temperature + top_p only; leave max_tokens/stop to the call site, where the right
value depends on the specific request.
"""
from __future__ import annotations

from typing import Any

# name -> the two knobs providers agree on. `chat(..., temperature=...)` in our
# LLM layer takes temperature directly; top_p is here as the documented companion
# and for callers/providers that pass it through.
PRESETS: dict[str, dict[str, float]] = {
    # Deterministic-ish: same input → (almost) same output. For extraction,
    # tool-calling, classification, math, code — anywhere a wrong token is a bug.
    "precise": {"temperature": 0.0, "top_p": 1.0},
    # The everyday default: coherent but not robotic. General chat and Q&A.
    "balanced": {"temperature": 0.7, "top_p": 0.95},
    # Wider, more surprising draws for ideation, naming, marketing copy, fiction.
    # Expect more misses — pair with validation or a human read.
    "creative": {"temperature": 1.1, "top_p": 0.98},
}

DEFAULT_PRESET = "balanced"


def preset(name: str = DEFAULT_PRESET) -> dict[str, Any]:
    """Return a *copy* of a named preset's sampling params.

    A copy, not the shared dict, so a caller can splice in `max_tokens`/`stop`
    (`preset("precise") | {"max_tokens": 256}`) without mutating the table for
    everyone else. Unknown names fail loudly rather than silently defaulting —
    a typo'd preset should surface, not quietly become "balanced".
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return dict(PRESETS[name])
