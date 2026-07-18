"""router.py — cost-aware routing: send each request to the cheapest model that can handle it.

The economic insight: most requests are easy. "What's the capital of France?",
"reword this sentence", "is this spam?" — a small local model answers those
perfectly. A minority are hard: multi-step reasoning, real code, dense math,
long analysis. If you send *everything* to a top-tier cloud model you pay
premium rates for the easy 90% you didn't need to. If you send everything to a
tiny local model, the hard 10% comes back wrong.

Routing splits the stream: the easy majority goes to a cheap/local model, the
hard minority to a strong one. Same bill, more work done — and lower latency on
the easy path, since local models answer without a network hop.

The heuristic here is intentionally cheap and readable: a few length and
keyword signals, no model call. It errs toward "harder" when unsure — a
false-hard costs money, but a false-easy costs a wrong answer, and the second is
worse. A production router might learn this from labelled traffic; the shape of
the decision is the same.
"""
from __future__ import annotations

import re

Provider = str  # "local" | "anthropic" | "openai"

# Words that signal the request needs real reasoning, code, or math — the kinds
# of task where a small model tends to fall over. Kept explicit so a reader can
# see exactly what "looks hard" means (and edit it).
_HARD_KEYWORDS = (
    "prove", "proof", "derive", "algorithm", "optimize", "optimise",
    "refactor", "debug", "stack trace", "traceback", "implement",
    "step by step", "step-by-step", "reason", "analyze", "analyse",
    "explain why", "trade-off", "tradeoff", "architecture", "complexity",
)

# Cheap structural tells that code or math is present.
_CODE_RE = re.compile(r"```|\bdef \b|\bclass \b|\bimport \b|[{};]|=>|\+\+|::")
_MATH_RE = re.compile(r"[∑∫√π≤≥≠]|\b\d+\s*[\^%]|\bmatrix\b|\bintegral\b|\bderivative\b")


def route(message: str) -> Provider:
    """Pick a provider for `message` by a cheap difficulty heuristic.

    Returns "local" for the easy majority, and a strong cloud model for hard
    cases — "anthropic" for reasoning/analysis-heavy prompts, "openai"
    otherwise. The split between the two strong providers is a stand-in for
    "route to whichever strong model you prefer for this shape of task"; swap it
    for your own preference.
    """
    text = message.strip()
    lower = text.lower()

    # Signals, each a reason to suspect the request is non-trivial.
    long = len(text) > 600                          # long prompts tend to be involved
    many_lines = text.count("\n") >= 8              # multi-part / structured asks
    has_code = bool(_CODE_RE.search(text))
    has_math = bool(_MATH_RE.search(text))
    keyword_hits = sum(kw in lower for kw in _HARD_KEYWORDS)

    hard_score = (
        long
        + many_lines
        + has_code
        + has_math
        + (keyword_hits >= 1)
        + (keyword_hits >= 3)  # several hard words compound — count them twice
    )

    if hard_score == 0:
        return "local"  # the easy 90%: keep it cheap and fast

    # Hard enough for a strong model. Reasoning/analysis-flavoured prompts and
    # plain math go to Anthropic; code and everything else to OpenAI. This is a
    # readable default, not a benchmark result — tune it to your own bill.
    reasoning_flavoured = has_math or keyword_hits >= 1
    if reasoning_flavoured and not has_code:
        return "anthropic"
    return "openai"
