"""LLM judge for answer correctness — used only to LABEL calibration data.

Conformal risk control needs a binary loss per calibration item: was answering
this question a mistake? For unanswerable questions that is decidable without a
judge (answering at all is the mistake). For answerable ones it needs someone to
compare the produced answer with the known one, which is what this does.

The judge is never in the serving path. It labels a calibration set offline; the
deployed gate is a single number compared against `support_score`. That matters,
because a judge in the loop would double the cost and put a second stochastic
component inside the guarantee it is meant to certify.

Known limitation, stated rather than hidden: the same model family judges and
answers, so a shared blind spot is invisible to this setup. The honest fix is a
different judge model, and the numbers should be read with that in mind.
"""

from __future__ import annotations

import re

from .llm import LLMClient

_SYSTEM = (
    "You compare a candidate answer against a reference answer for the same "
    "question. Reply with exactly one word:\n"
    "  CORRECT    the candidate states the same fact as the reference, in any wording\n"
    "  INCOMPLETE the candidate is not wrong but omits the key fact\n"
    "  WRONG      the candidate contradicts the reference or invents something\n"
    "Ignore style, length and citation markers. Judge the factual content only."
)

_VERDICTS = ("CORRECT", "INCOMPLETE", "WRONG")


def judge_answer(question: str, candidate: str, reference: str, llm: LLMClient) -> str:
    """Return CORRECT / INCOMPLETE / WRONG (defaults to WRONG if unparseable)."""
    if not candidate.strip():
        return "WRONG"
    prompt = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Candidate answer: {candidate}\n\nVerdict:"
    )
    text = (llm.complete(_SYSTEM, prompt).text or "").upper()
    for v in _VERDICTS:
        if re.search(rf"\b{v}\b", text):
            return v
    return "WRONG"


def loss_from_verdict(verdict: str) -> float:
    """Binary loss for conformal risk control.

    INCOMPLETE counts as a loss. For a maintenance assistant an answer that
    omits the key fact is a failure, not a partial credit — the technician acts
    on what they were told.
    """
    return 0.0 if verdict == "CORRECT" else 1.0
