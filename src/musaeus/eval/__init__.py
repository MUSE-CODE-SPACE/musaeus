"""eval/ — how do you know your agent got *better* and not just *different*?

This module is Musaeus's answer to a question every LLM app eventually faces:
"I changed a prompt / model / temperature — did that help or hurt?" You cannot
answer that by eyeballing one reply. You need an **offline eval set**: a fixed
list of cases (question + what a good answer looks like) that you re-run on every
change, the way a test suite guards a codebase against regressions. Same inputs,
same yardstick, every time — so a score that drops is a real signal, not vibes.

Scoring free-form text is the hard part. Two families of methods:
  - *Reference-based* metrics (exact match, BLEU, ROUGE) compare against a gold
    string. Cheap and deterministic, but brittle: a correct paraphrase scores 0.
  - *LLM-as-judge*: ask a capable model to grade the answer against a rubric.
    Flexible and close to human judgement — and what we implement here.

LLM-as-judge is powerful but **biased**, and a reader should know its failure
modes rather than trust the number blindly:
  - *Position bias*: in A-vs-B comparisons a judge favors whichever answer came
    first. Mitigate by swapping order and averaging. (We grade one answer at a
    time, which sidesteps this, but pairwise setups must handle it.)
  - *Verbosity bias*: judges reward longer, more confident-sounding answers even
    when they add nothing. Rubrics should reward correctness, not word count.
  - *Self-preference*: a model tends to rate its own outputs higher, so judging
    with the same model you're testing quietly inflates scores.
  - *Leniency / clustering*: judges drift toward the middle (all 3s and 4s). A
    tight 1-5 rubric with explicit anchors fights this.
The judge is a useful proxy, not ground truth — track its scores as a trend and
spot-check real answers by hand.
"""
from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any, Callable

from ..llm import LLM

# The judge is told exactly how to score and, crucially, to answer in a fixed
# JSON shape — structure we can parse instead of hoping a number is in the prose.
_JUDGE_SYSTEM = (
    "You are a strict, fair evaluator. Grade the ANSWER to the QUESTION against "
    "the RUBRIC on an integer scale of 1 to 5, where 1 = badly wrong/off-topic "
    "and 5 = fully correct and complete. Judge substance, not length or "
    "confidence; a short correct answer beats a long vague one. Reply with ONLY "
    'a JSON object: {"score": <int 1-5>, "rationale": "<one sentence why>"}.'
)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the judge's JSON out of whatever it actually returned.

    Models wrap JSON in prose or ```json fences even when told not to, so we grab
    the first {...} block rather than json.loads() the whole string. Robust
    parsing is not optional here: a judge that occasionally chats before its JSON
    should not crash a 200-case eval run.
    """
    # Greedy `.*` on purpose: a lazy match stops at the first `}` and mangles
    # nested objects.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return {}


def _clamp_score(value: Any) -> int:
    """Coerce whatever the judge put in "score" to an int in [1, 5].

    Defends against "4/5", "4.0", "four" (falls back to 1), and out-of-range
    numbers — the aggregate mean must never be poisoned by a stray value.
    """
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        m = re.search(r"[1-5]", str(value))
        n = int(m.group(0)) if m else 1
    return max(1, min(5, n))


def judge(llm: LLM, question: str, answer: str, rubric: str) -> dict:
    """Ask a judge model to grade one answer. Returns {score:int, rationale:str}.

    `llm` is any Musaeus LLM — ideally a *different* (stronger) model than the one
    that produced `answer`, to avoid the self-preference bias noted above.
    temperature=0 keeps grading as reproducible as sampling allows: you want the
    same answer to earn the same score across runs.
    """
    user = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"RUBRIC (what a good answer must contain):\n{rubric}"
    )
    reply = llm.chat(
        [{"role": "system", "content": _JUDGE_SYSTEM},
         {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=256,
    )
    parsed = _extract_json(reply.text)
    return {
        "score": _clamp_score(parsed.get("score", 1)),
        "rationale": str(parsed.get("rationale", "")).strip() or "(no rationale returned)",
        "answer": answer,
    }


def run_eval(cases: list[dict], answer_fn: Callable[[str], str], *, llm: LLM) -> dict:
    """Run a whole offline eval set and aggregate the scores.

    Each case is {"question": str, "rubric": str}. For every case we:
      1. call `answer_fn(question)` — the system under test (an Agent, a raw LLM,
         a RAG pipeline; eval doesn't care, it only sees text in / text out), then
      2. judge that answer against the case's rubric.

    Returns {"mean_score": float, "n": int, "results": [<per-case dicts>]}. Keep
    the per-case results: an aggregate that dropped is only actionable once you can
    see *which* cases regressed and read the judge's rationale for each.

    `llm` is keyword-only to make call sites read as "…, llm=judge_model" — a loud
    reminder that the judge is a deliberate choice, not the model being graded.
    """
    results: list[dict] = []
    for case in cases:
        question, rubric = case["question"], case["rubric"]
        try:
            answer = answer_fn(question)
        except Exception as exc:  # a broken answer_fn scores 1, not aborts the run
            results.append({"question": question, "score": 1, "answer": "",
                            "rationale": f"answer_fn raised: {exc}"})
            continue
        verdict = judge(llm, question, answer, rubric)
        results.append({"question": question, **verdict})

    scores = [r["score"] for r in results]
    return {
        "mean_score": round(mean(scores), 3) if scores else 0.0,
        "n": len(results),
        "results": results,
    }
