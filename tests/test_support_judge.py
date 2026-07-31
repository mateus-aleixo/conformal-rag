"""Support score and judge — deterministic behaviour, no model required."""

import pytest

from conformal_rag.answer import answer
from conformal_rag.conformal import ConformalGate
from conformal_rag.judge import judge_answer, loss_from_verdict
from conformal_rag.llm import StubLLM
from conformal_rag.retrieve import retrieve
from conformal_rag.support import support_score


def test_support_parses_a_bare_integer(store, embedder):
    hits = retrieve(store, embedder, "oil pressure", k_final=3)
    assert support_score("q", hits, StubLLM(script=["85"])) == 0.85
    assert support_score("q", hits, StubLLM(script=["0"])) == 0.0
    assert support_score("q", hits, StubLLM(script=["100"])) == 1.0


def test_support_survives_chatty_and_broken_replies(store, embedder):
    hits = retrieve(store, embedder, "oil pressure", k_final=3)
    # a number embedded in prose is still usable
    assert support_score("q", hits, StubLLM(script=["Score: 70 out of 100"])) == 0.70
    # no number at all -> no support, so the gate abstains rather than guesses
    assert support_score("q", hits, StubLLM(script=["I cannot say"])) == 0.0
    # out-of-range values are clamped, not trusted
    assert support_score("q", hits, StubLLM(script=["150"])) == 1.0
    assert support_score("q", hits, StubLLM(script=["-20"])) == 0.0


def test_support_is_zero_without_hits():
    assert support_score("q", [], StubLLM(script=["100"])) == 0.0


def test_support_prompt_fences_the_excerpts(store, embedder):
    """Retrieved text must reach the judge as data, like everywhere else."""
    hits = retrieve(store, embedder, "filter service hours", k_final=3)
    llm = StubLLM(script=["50"])
    support_score("q", hits, llm)
    _, user = llm.calls[0]
    assert "<untrusted-data>" in user


@pytest.mark.parametrize("reply,expected", [
    ("CORRECT", "CORRECT"),
    ("the candidate is INCOMPLETE", "INCOMPLETE"),
    ("WRONG", "WRONG"),
    ("banana", "WRONG"),          # unparseable defaults to a loss
])
def test_judge_verdicts(reply, expected):
    assert judge_answer("q", "candidate", "reference", StubLLM(script=[reply])) == expected


def test_judge_treats_empty_answer_as_wrong():
    assert judge_answer("q", "   ", "reference", StubLLM(script=["CORRECT"])) == "WRONG"


def test_incomplete_counts_as_a_loss():
    """A maintenance answer missing the key fact is a failure, not partial credit."""
    assert loss_from_verdict("CORRECT") == 0.0
    assert loss_from_verdict("INCOMPLETE") == 1.0
    assert loss_from_verdict("WRONG") == 1.0


def test_gate_can_run_on_the_support_signal(store, embedder, cfg):
    """use_support routes the gate through support_score instead of RRF agreement."""
    gate = ConformalGate(alpha=0.1)
    gate.global_threshold = 0.9
    # judged unsupported (10) -> below threshold -> abstain, and no answer is generated
    llm = StubLLM(script=["10", "SHOULD NOT BE REACHED"])
    ans = answer("anything", store, embedder, llm, cfg, gate=gate, use_support=True)
    assert ans.abstained and ans.reason == "gate"
    assert ans.confidence == 0.10
    assert len(llm.calls) == 1          # scored once, never asked to answer

    # judged supported (95) -> above threshold -> answers
    llm2 = StubLLM(script=["95", "The answer is X [1]."])
    ans2 = answer("anything", store, embedder, llm2, cfg, gate=gate, use_support=True)
    assert not ans2.abstained
    assert ans2.confidence == 0.95
    assert "[1]" in ans2.text
