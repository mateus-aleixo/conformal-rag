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

## Still to come

- **M1** — retrieval recall@5 on the manually verified golden set (not the seeds).
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
