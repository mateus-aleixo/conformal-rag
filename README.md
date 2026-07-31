# conformal-rag

**Retrieval-augmented question answering over industrial maintenance manuals that
knows when to shut up.** Instead of hallucinating on questions the corpus cannot
answer, the system **abstains with a calibrated, distribution-free guarantee** on its
error rate — conformal risk control applied to selective question answering.

Sibling of [conformal-rul](https://github.com/mateus-aleixo/conformal-rul) (remaining
useful life with calibrated intervals, live on AWS Lambda). Same thesis, new modality:
*a prediction without a trustworthy confidence statement is not a decision aid.* The
agent in this repo calls the live conformal-rul API as one of its tools.

> **Status: retrieval and answering measured; the conformal gate is not built yet.**
> This README was pushed before any code, on purpose, and the git log is the honest
> record. Where it stands, on 20 manually grounded questions over 1,752 real chunks
> ([`docs/results.md`](docs/results.md)):
>
> | | |
> |---|---|
> | recall@5 (bge) | **1.00** — 0.93 with the deterministic CI embedder |
> | refusal on unanswerable questions | **5/5** |
> | invalid citations | **0 / 20 responses** |
>
> The headline finding is a negative one, and it changed the design: **better
> retrieval made abstention *harder*.** A stronger retriever confidently finds
> something plausible even when the corpus cannot answer, so the answerable /
> unanswerable confidence gap halved. The signal that did separate them cleanly was
> the model reading the excerpts — so the conformal gate is being built on the
> generation step, not the retrieval step. Five questions is five questions, not a
> guarantee; the caveat is spelled out in the results.

## Why abstention, and why conformal

RAG systems fail worst on the questions they cannot answer: retrieval returns
something vaguely related, the LLM writes a fluent paragraph, and the user has no way
to know it is wrong. Confidence heuristics ("the model seemed unsure") carry no
guarantee.

Conformal risk control does. Given a calibration set of questions labelled
answered-correctly / answered-wrongly, choose the confidence threshold

λ̂ = inf { λ : (n/(n+1)) · R̂(λ) + 1/(n+1) ≤ α }

and answer only above it. The result, under exchangeability: **the wrong-answer rate
among answered questions is ≤ α**, finite-sample, no distributional assumptions
(Angelopoulos et al., 2022; Mohri & Hashimoto, 2024). A Mondrian split gives
per-question-type thresholds with a small-group fallback — the same construction used
in conformal-rul for operating regimes.

## Architecture

```
PDFs ──ingest──> chunks ──> SQLite (FTS5 BM25 + embedded vectors)
                                    │
question ──> hybrid retrieval (RRF) ──> rerank (trained cross-encoder)   [M4]
                                    │
                             conformal gate ──abstain──> "cannot answer, here's why"
                                    │ answer
                             LLM with citations ──> JSONL trace (tokens, latency, cost)
                                    ▲
        agent loop (from scratch): search_docs · predict_rul (live API) · calculator
```

Design choices, deliberately boring where boring is right:

- **Framework-free.** The agent loop, tool protocol, guardrails and conformal gate are
  ~a few hundred lines of plain Python. No LangChain. When behaviour is wrong, the
  bug is in this repo, findable.
- **SQLite for everything at rest.** FTS5 gives BM25; embeddings live as blobs and are
  searched with NumPy (the corpus is thousands of chunks, not billions — brute force
  is milliseconds and has no failure modes). One file, deploys anywhere.
- **Provider-agnostic LLM client.** Local [Ollama](https://ollama.com) by default; any
  OpenAI-compatible endpoint via env vars; a deterministic stub for tests, so CI runs
  the full suite with no model downloads and no API spend.
- **Security is tested, not claimed.** `evals/injection_cases.jsonl` embeds
  adversarial instructions inside corpus documents; CI asserts the system treats
  retrieved text as data, not as orders.

## Corpus

Public-domain **US Army technical manuals** (works of the US federal government,
17 U.S.C. §105) — e.g. TM 9-8000 *Principles of Automotive Vehicles*. Real, messy,
industrial PDFs. `scripts/fetch_corpus.py` downloads them; raw PDFs stay out of git.

## Quickstart

```bash
uv sync --all-extras          # or: pip install -e ".[dev]"
uv run pytest                 # full suite, no network, no models
uv run python -m conformal_rag ingest data/raw/*.pdf
uv run python -m conformal_rag ask "What does low oil pressure at idle indicate?"
uv run python -m conformal_rag agent "Remaining life for these engine readings: ..."
```

## Roadmap (dates are the plan, the git log is the truth)

| Milestone | Window | Deliverable | |
|---|---|---|---|
| M0 | Aug 1–3 | README, scaffold, CI, corpus fetcher | ✅ |
| M1 | Aug 4–15 | Ingestion → hybrid retrieval; grounded golden set; recall measured | ✅ |
| M2 | Aug 16–27 | Cited answers, provider client, tracing; refusal + citation eval | ✅ |
| M3 | Sep 1–8 | Agent + tools incl. the live conformal-rul API; injection suite in CI | ✅ |
| M4 | Sep 12–22 | Conformal gate on the **generation** signal + held-out risk plot; **trained reranker** with before/after table | |
| M5 | Sep 23–30 | Evals hardened, Docker; v1 | |

M0–M3 landed ahead of the plan because the scaffold turned out to carry most of
M2 and M3 already; the dates above are left unedited so the schedule can be
compared against what actually happened. **M4 is the substance** — everything so
far is the apparatus it needs.

Non-goals for v1: UI, multi-corpus, fine-tuned generator, Kubernetes, streaming.

## References

- Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control*, 2022.
- Mohri, Hashimoto — *Language Models with Conformal Factuality Guarantees*, 2024.
- Barber, Candès, Ramdas, Tibshirani — *Predictive inference with the jackknife+*, 2021.

MIT licence. Built by [Mateus Aleixo](https://github.com/mateus-aleixo).
