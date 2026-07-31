from conformal_rag.answer import answer, build_prompt, confidence_from_hits
from conformal_rag.conformal import ConformalGate
from conformal_rag.llm import StubLLM
from conformal_rag.retrieve import retrieve


def test_answer_includes_citations_and_sources(store, embedder, cfg):
    llm = StubLLM(script=["Low oil pressure indicates a worn pump [1]."])
    ans = answer("What does low oil pressure at idle indicate?", store, embedder, llm, cfg)
    assert not ans.abstained
    assert "[1]" in ans.text
    assert ans.hits[0].doc == "tm-engine.pdf"
    # prompt carried fenced excerpts, not raw text
    system, user = llm.calls[0]
    assert "<untrusted-data>" in user
    assert "INSUFFICIENT EVIDENCE" in system


def test_gate_forces_abstention(store, embedder, cfg):
    gate = ConformalGate(alpha=0.1)
    gate.global_threshold = 2.0  # unattainable → always abstain
    llm = StubLLM()
    ans = answer("anything at all", store, embedder, llm, cfg, gate=gate)
    assert ans.abstained and ans.reason == "gate"
    assert llm.calls == []  # abstention must not spend model tokens


def test_confidence_zero_without_hits():
    assert confidence_from_hits([]) == 0.0


def test_confidence_bounded(store, embedder, cfg):
    hits = retrieve(store, embedder, "oil pressure", k_final=3)
    c = confidence_from_hits(hits)
    assert 0.0 <= c <= 1.0


def test_build_prompt_names_doc_and_page(store, embedder, cfg):
    hits = retrieve(store, embedder, "thermostat coolant", k_final=2)
    prompt = build_prompt("q", hits)
    assert "tm-engine.pdf" in prompt and "p.13" in prompt
