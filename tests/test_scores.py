"""Candidate nonconformity scores — deterministic behaviour, no model needed."""

import pytest

from conformal_rag.llm import StubLLM
from conformal_rag.retrieve import retrieve
from conformal_rag.scores import (
    agreement,
    content_words,
    groundedness_score,
    self_consistency_score,
    support_score_v2,
)


def test_agreement_bounds():
    assert agreement(["the pump is worn", "the pump is worn"]) == 1.0
    assert agreement(["worn oil pump", "battery electrolyte gravity"]) == 0.0
    assert 0.0 < agreement(["worn oil pump", "worn oil pump bearing"]) < 1.0


def test_agreement_needs_two_usable_samples():
    """One sample, or blanks, cannot show agreement — must not read as certainty."""
    assert agreement(["only one"]) == 0.0
    assert agreement([]) == 0.0
    assert agreement(["something", "   "]) == 0.0


def test_agreement_ignores_stopwords():
    """Two unrelated answers must not look similar for both being English."""
    a = "It is in the engine and it is on the block"
    b = "It is at the radiator and it is by the core"
    assert agreement([a, b]) == 0.0
    assert content_words("The pump is in the engine") == {"pump", "engine"}


def test_groundedness_parses_and_clamps(store, embedder):
    hits = retrieve(store, embedder, "oil pressure", k_final=3)
    assert groundedness_score("some answer", hits, StubLLM(script=["73"])) == 0.73
    assert groundedness_score("some answer", hits, StubLLM(script=["140"])) == 1.0
    assert groundedness_score("some answer", hits, StubLLM(script=["no idea"])) == 0.0


def test_groundedness_zero_without_answer_or_hits(store, embedder):
    hits = retrieve(store, embedder, "oil pressure", k_final=3)
    assert groundedness_score("   ", hits, StubLLM(script=["100"])) == 0.0
    assert groundedness_score("an answer", [], StubLLM(script=["100"])) == 0.0


def test_groundedness_fences_the_excerpts(store, embedder):
    hits = retrieve(store, embedder, "filter service hours", k_final=3)
    llm = StubLLM(script=["50"])
    groundedness_score("an answer", hits, llm)
    _, user = llm.calls[0]
    assert "<untrusted-data>" in user


def test_self_consistency_returns_score_and_samples():
    llm = StubLLM(script=["worn oil pump", "worn oil pump", "worn oil pump"])
    score, samples = self_consistency_score("sys", "prompt", llm, k=3)
    assert len(samples) == 3
    assert score == 1.0
    assert len(llm.calls) == 3


def test_support_v2_prompt_drops_the_anchors(store, embedder):
    """M4 collapsed to four values because the prompt offered 0/50/100 as examples."""
    hits = retrieve(store, embedder, "oil pressure", k_final=3)
    llm = StubLLM(script=["43"])
    assert support_score_v2("q", hits, llm) == 0.43
    system, _ = llm.calls[0]
    assert "avoid round numbers" in system
    assert "50  partially covered" not in system      # the v1 anchor block is gone


@pytest.mark.parametrize("reply,expected", [("0", 0.0), ("100", 1.0), ("-5", 0.0)])
def test_support_v2_clamps(store, embedder, reply, expected):
    hits = retrieve(store, embedder, "oil pressure", k_final=3)
    assert support_score_v2("q", hits, StubLLM(script=[reply])) == expected


# ------------------------------------------------- head-purity combination

def test_head_score_is_conjunctive():
    """min, not mean: one strong signal must not carry a weak one into the head."""
    from conformal_rag.scores import head_score
    assert head_score({"a": 0.9, "b": 0.1}) == 0.1        # weakest link wins
    assert head_score({"a": 0.9, "b": 0.8}) == 0.8
    assert head_score({}) == 0.0


def test_head_score_vetoes_refusal_and_bad_citations():
    from conformal_rag.scores import head_score
    hi = {"a": 1.0, "b": 1.0}
    assert head_score(hi, "INSUFFICIENT EVIDENCE", 5) == 0.0
    assert head_score(hi, "The answer is X [9].", 5) == 0.0   # excerpt 9 never supplied
    assert head_score(hi, "The answer is X [2].", 5) == 1.0   # in range
    assert head_score(hi, "The answer is X [9].", 5, veto=False) == 1.0


def test_head_score_clamps():
    from conformal_rag.scores import head_score
    assert head_score({"a": 1.5, "b": 2.0}) == 1.0
    assert head_score({"a": -0.5, "b": 0.5}) == 0.0


def test_consistency_excluding_refusals():
    """Three identical refusals are perfect agreement and must NOT score 1.0."""
    from conformal_rag.scores import consistency_excluding_refusals
    refusals = ["INSUFFICIENT EVIDENCE"] * 3
    assert consistency_excluding_refusals(refusals) == 0.0
    mixed = ["worn oil pump", "INSUFFICIENT EVIDENCE", "worn oil pump"]
    assert consistency_excluding_refusals(mixed) == 0.0
    real = ["worn oil pump", "worn oil pump", "worn oil pump"]
    assert consistency_excluding_refusals(real) == 1.0
