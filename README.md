# conformal-rag

[![ci](https://github.com/mateus-aleixo/conformal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/mateus-aleixo/conformal-rag/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Retrieval-augmented question answering over industrial maintenance manuals that
knows when to shut up.** Instead of hallucinating on questions the corpus cannot
answer, the system **abstains with a calibrated, distribution-free guarantee** on its
error rate — conformal risk control applied to selective question answering.

Third of a series — one thesis, three modalities: *a prediction without a trustworthy
confidence statement is not a decision aid.*

| repo | modality | the guarantee |
|---|---|---|
| [conformal-rul](https://github.com/mateus-aleixo/conformal-rul) | sensor sequences | RUL intervals with verified coverage, live on AWS Lambda |
| [conformal-seg](https://github.com/mateus-aleixo/conformal-seg) | vision | defect masks bounding the missed-defect rate |
| **conformal-rag** | language | selective QA that abstains at a calibrated error rate |

The agent in this repo calls the **live conformal-rul API** as one of its tools, so the
series composes rather than merely rhyming.

> **Status: gate built, calibrated, and then improved once its ceiling was found.**
> Full numbers in [`docs/results.md`](docs/results.md); 100 questions, threshold
> fitted on one half and every figure reported from the other.
>
> | risk being bounded | ungated | gated | |
> |---|---|---|---|
> | answered a question the corpus **cannot** answer | 0.260 | **0.031**, answering 64% | ✅ α = 0.1 |
> | **any** mistake (unanswerable *or* wrong answer) | 0.540 | **0.312**, answering 64% | ⚠️ α = 0.4, not 0.2 |
>
> The reason is that a conformal gate can only bound a risk its score can *rank*, and
> the first score judged the excerpts rather than the answer. Replacing it with one
> that looks at what the model actually said — agreement across sampled answers,
> combined with whether the answer is grounded in its citations — raises the ability
> to rank correctness from **AUC 0.51 (a coin flip) to 0.75**, and pulls the
> any-mistake risk from 0.540 down to **0.312 while still answering 64%**.
>
> **α = 0.2 is reached at 14 B** — and not for the reason predicted. Every generator
> re-judged by the same judge:
>
> | generator | base error | support on *unanswerable* | best α met | coverage |
> |---|---|---|---|---|
> | 3 B | 0.427 | 0.080 | none | — |
> | 7 B | **0.347** | 0.060 | 0.40 | 74% |
> | 14 B | 0.373 | **0.020** | **0.20** | 30% |
>
> The 14 B answers *slightly worse* than the 7 B and gates far better. What scale
> improved was not accuracy but the model's judgement about its evidence — it is much
> better at recognising when the excerpts cannot answer the question, and a conformal
> gate converts exactly that into a guarantee. **Scaling bought calibration, not
> correctness.**
>
> The price is coverage: the gate meets its target by declining seven questions in ten.
> Whether that is the right trade depends on whether a wrong maintenance answer is
> worse than no answer. Here it is.
>
> **The sharpest finding is about how to choose a score.** A combined
> groundedness × self-consistency signal ranks correctness far better —
> **AUC 0.845 vs 0.697** — and *cannot* meet α = 0.2 at any threshold, while the
> weaker-ranking score can. AUC describes the whole ordering; a conformal gate draws
> one line and keeps what is above it, so the only thing that matters is whether some
> **top bucket is nearly pure**. Pick a nonconformity score by the purity of its head
> at the α you need, not by AUC — selecting on AUC picks the score that cannot deliver
> the guarantee. (At a looser α = 0.30 the combined score is the right pick: 60%
> coverage against 34%.)
>
> One methodological note that cost a wrong conclusion before it was caught: swapping
> model swaps the **judge** too, and judges vary *unpredictably* — the 7 B judge is
> stricter than both the 3 B and the 14 B. Hold the judge fixed when comparing
> generators.

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
question ──> hybrid retrieval (RRF) ──> [rerank: measured, not adopted — see results]
                                    │
                             LLM with citations ──> JSONL trace (tokens, latency, cost)
                                    │
                    groundedness × self-consistency  =  nonconformity score
                                    │
                             conformal gate ──abstain──> "cannot answer, here's why"
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
| M4 | Sep 12–22 | Conformal gate on the **generation** signal + held-out risk plot; reranker **measured and rejected** (+0.02 recall@5 for +899 ms) | ✅ |
| M5 | Sep 23–30 | Correctness-aware score: AUC 0.51 → **0.75**, risk 0.540 → **0.312** at 64% coverage | ✅ |
| — | Jul 31 | 7 B tested with the judge held fixed: −27% error, 74% coverage, α = 0.2 unreachable | ✅ |
| — | Aug 1 | 14 B: **α = 0.2 met** (risk 0.133) at 30% coverage — via calibration, not accuracy | ✅ |
| — | Aug 1 | Combined score on 14 B: best AUC (0.845), **worse gate** — head purity ≠ ranking quality | ✅ |
| next | — | A score built for head purity directly; a harder, hand-written golden set | |

M0–M3 landed ahead of the plan because the scaffold carried most of M2 and M3
already; the dates are left unedited so the schedule can be compared with what
happened. Two planned items were **measured and then dropped** — the trained
reranker (bought +0.02 recall@5 for +899 ms) and the anchor-free support prompt
(fixed the granularity, not the blindness). Both are written up rather than
quietly removed.

Non-goals for v1: UI, multi-corpus, fine-tuned generator, Kubernetes, streaming.

## References

- Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control*, 2022.
- Mohri, Hashimoto — *Language Models with Conformal Factuality Guarantees*, 2024.
- Barber, Candès, Ramdas, Tibshirani — *Predictive inference with the jackknife+*, 2021.

MIT licence. Built by [Mateus Aleixo](https://github.com/mateus-aleixo).
