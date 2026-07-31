# Results

**Nothing to report yet — honestly.** This file is the single place metrics land,
in the same pattern as conformal-rul's `docs/results.md`. Numbers appear here only
when produced by `uv run python -m conformal_rag calibrate` / the eval harness on
the real corpus, with the commit hash that produced them.

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
