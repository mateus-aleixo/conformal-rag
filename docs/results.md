# Results

**No benchmark numbers yet — honestly.** Retrieval recall, answer correctness and
abstention risk require the manually verified golden set (M1/M4); until then this
file records only what has actually been run, which is the pipeline working
end to end.

## Verified end to end, 2026-07-31

Corpus: 1,752 chunks from two public-domain US Army technical manuals.
Model: `qwen2.5:3b-instruct` via local Ollama. Retrieval: hybrid BM25 + vectors,
RRF-fused.

**Grounded answer with a citation.** *"What is the purpose of the cooling system
thermostat in an engine?"* →

> The thermostat in an engine's cooling system regulates engine temperature by
> controlling the amount of coolant flowing from the engine block to the radiator
> core. It operates based on heat, and if it fails, it will be in the opened
> position so as to allow the free circulation of coolant through the engine **[4]**.

with `[4]` resolving to TM-9-8000 p.241. All five hits found by **both** retrievers.

**Correct refusal.** *"What should be inspected before starting the generator
set?"* → `INSUFFICIENT EVIDENCE`. Retrieval surfaced only the automotive manual
(the generator document in the corpus is a parts catalogue with no pre-start prose),
and the system declined rather than improvising from adjacent text. This is the
intended behaviour and it happened without the conformal gate even being fitted —
the gate makes the refusal rate *calibrated*, the prompt makes it *possible*.

**Agent tool loop.** `calculator` → `17 * 23` → `391`, one step, clean JSON protocol.

**Cross-project composition.** `predict_rul` called the live
[conformal-rul](https://github.com/mateus-aleixo/conformal-rul) service on AWS
Lambda with a real C-MAPSS cycle:

```json
{"rul_cycles": 118.9,
 "interval": {"lower": 87.5, "upper": 125.0, "coverage": 90},
 "risk_band": "healthy", "model": "transformer"}
```

## M1 — retrieval, measured 2026-07-31

Golden set: 20 questions (15 answerable, 5 unanswerable), written by **reading the
source chunks**, not by asking the retriever where it would look — grounding gold on
the retriever's own top hit would only measure its tie-breaking. Scored at page level
with ±1 slack, because chunks overlap and a boundary can split an answer.
Corpus: 1,752 chunks. `python scripts/eval_retrieval.py --embedder {hash,bge}`

| embedder | recall@5 | recall@1 | MRR | top hit found by both retrievers |
|---|---|---|---|---|
| `hash` (deterministic placeholder, CI default) | 0.93 | 0.73 | 0.851 | 18/20 |
| **`bge-small-en-v1.5`** | **1.00** | **0.93** | **0.950** | 19/20 |

Real embeddings fix the one miss — a "what keeps the brakes working if the power
steering fails?" question that the lexical path answered with the wrong page while
reporting **high** confidence (0.969). Hybrid retrieval is doing real work either
way: the top hit was found by *both* BM25 and the vector search on 18–19 of 20
questions, so this is not BM25 with extra steps.

### The finding that matters, and it is not the recall

| embedder | mean confidence, answerable | unanswerable | **gap** |
|---|---|---|---|
| `hash` | 0.946 | 0.643 | **0.303** |
| `bge` | 0.992 | 0.831 | **0.161** |

**Better retrieval made the abstention signal worse.** Recall went up and the gap
between answerable and unanswerable questions roughly halved — because a stronger
retriever confidently finds *something* topically plausible even when the corpus
cannot answer the question. Retrieval agreement measures "did my retrievers concur",
which is not the same as "is the answer in here".

That is a problem for M4, and a useful one: it means the conformal gate cannot be
built on retrieval confidence alone, and the nonconformity score needs a
judge-based signal. Better to learn that from a 20-question eval than from a
calibration that quietly certifies confident nonsense. The unanswerable questions
were written to be *plausible* (a torque spec for an engine the corpus does not
cover) rather than absurd, which is precisely why they are hard to separate.

## M2 — answers, measured 2026-07-31

Same 20 questions through the full pipeline. `qwen2.5:3b-instruct` via local
Ollama, bge retrieval. `python scripts/eval_answers.py --provider ollama`

Three things are scored, all decidable without a judge model:

| | | |
|---|---|---|
| **Refusal on unanswerable** | **5 / 5 = 1.00** | said `INSUFFICIENT EVIDENCE` rather than improvising |
| Answer rate on answerable | 14 / 15 = 0.93 | |
| **Invalid citations** | **0 / 20 responses** | every `[n]` indexed a excerpt that was actually supplied |
| Answers carrying a citation | 14 / 14 = 1.00 | |
| Median latency | 3.7 s | 3 B model, CPU-class hardware |

Semantic correctness — "is the answer *right*" — is deliberately **not** scored
here. That needs a judge model and a rubric (M4/M5). Approximating it with string
overlap would produce a number that looks like accuracy and isn't.

### This answers the question M1 raised

M1 found that better retrieval *shrank* the confidence gap between answerable and
unanswerable questions (0.303 → 0.161), which made retrieval agreement look like a
poor basis for abstention. M2 shows where the signal actually lives: **the model
reading the excerpts separated them perfectly, 5 out of 5, where retrieval
confidence could not.**

So the M4 nonconformity score should be built on the generation step, not the
retrieval step. That is a design decision now supported by evidence rather than by
taste — and it is the opposite of what the v0 confidence heuristic assumed.

**Sample-size caveat, stated plainly:** 5/5 on five questions is not a 100% refusal
rate. It is "no failures observed in five attempts", whose 95% upper bound is
roughly 45%. The number to trust is the *direction*, not the value. Expanding the
unanswerable set is the first task of M4, precisely because the gate will be
calibrated against it.

### The one over-refusal

`g-05` — *"Why does damping on rebound only force the use of stiffer springs?"* —
was refused despite the answer being on the retrieved page. It is the only
**reasoning**-type question that required combining two sentences rather than
quoting one. The system errs conservative, which is the right direction for a
maintenance assistant, but it marks a real ceiling: strict grounding prompts
suppress inference, and questions needing a short chain of reasoning are where
that costs recall.

## Still to come

- **M1/M2 extension** — more golden questions; 20 is enough to expose a direction,
  not to publish a number with a confidence interval.
- **M4** — reranker before/after: recall@5 without vs with the trained cross-encoder.
- **M4** — abstention: held-out selective risk vs α, answer rate, Mondrian breakdown.
- **M5** — answer correctness (judge + exact-match subset), cost and latency by provider.

Planned tables (see README roadmap):

- **M1** — retrieval recall@5 on the manually verified golden set (not the seeds).
- **M4** — reranker before/after: recall@5 without vs with the trained
  cross-encoder, same split, same seed.
- **M4** — abstention: held-out selective risk vs α, answer rate, per-group
  (Mondrian) breakdown, and the risk plot.
- **M5** — end-to-end answer correctness (LLM-judge + exact-match subset), cost
  and latency per question by provider.

Rule carried over from conformal-rul: the headline is whatever the data says,
including when the boring baseline wins.
