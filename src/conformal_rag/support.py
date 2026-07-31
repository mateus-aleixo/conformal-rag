"""Generation-side support score — the conformal gate's nonconformity signal.

M1 measured retrieval-agreement confidence and found it a poor abstention
signal: a *better* retriever shrank the gap between answerable and unanswerable
questions (0.303 → 0.161), because a strong retriever confidently finds
something topically plausible either way. M2 then showed the model reading the
excerpts separated them cleanly. So the score the gate calibrates on is taken
here, at the generation step, not from retrieval.

The model is asked one narrow question — *do these excerpts contain the answer?*
— and made to reply with a bare integer. That is deliberately not "how confident
are you in your answer", which invites the model to rate its own prose; it is a
question about the evidence, which it can actually inspect.

Ollama is asked for `temperature: 0`, so the score is reproducible for a fixed
model and corpus.
"""

from __future__ import annotations

import re

from .guard import SYSTEM_RULES, fence
from .llm import LLMClient
from .retrieve import Hit

_SYSTEM = (
    "You judge whether a set of excerpts contains the information needed to answer "
    "a question. You do NOT answer the question. Reply with a single integer from "
    "0 to 100 and nothing else.\n"
    "  0   the excerpts are unrelated, or discuss the topic without stating the fact asked for\n"
    "  50  partially covered: some of what is needed is present, some is missing\n"
    "  100 the excerpts plainly and completely contain the answer\n"
    "Judge only what is written. Do not use outside knowledge. " + SYSTEM_RULES
)

_INT = re.compile(r"-?\d+")


def support_score(question: str, hits: list[Hit], llm: LLMClient) -> float:
    """Fraction in [0, 1] estimating whether `hits` can answer `question`.

    Returns 0.0 when there is nothing to judge or the model does not produce a
    number — an unparseable judgement is treated as no support, so the failure
    mode is abstention rather than a confident answer.
    """
    if not hits:
        return 0.0
    excerpts = "\n\n".join(
        f"[{i}] ({h.doc}, p.{h.page})\n{fence(h.text)}" for i, h in enumerate(hits, start=1)
    )
    result = llm.complete(_SYSTEM, f"Question: {question}\n\nExcerpts:\n\n{excerpts}\n\nScore:")
    m = _INT.search(result.text or "")
    if not m:
        return 0.0
    return max(0.0, min(100.0, float(m.group(0)))) / 100.0
