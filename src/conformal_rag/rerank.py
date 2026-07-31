"""Cross-encoder reranking over the fused candidate list.

Retrieval scores a question and a chunk *separately* and compares vectors; a
cross-encoder reads the pair together, which is slower but far better at
telling a chunk that shares vocabulary from one that actually answers.
So: retrieve wide and cheaply, rerank narrow and expensively.

**Pretrained, not fine-tuned, and that is a deliberate call.** The plan called
for a trained reranker. The golden set has 60 answerable questions — roughly a
few hundred query/passage pairs. Fine-tuning a cross-encoder on that would
overfit and produce a number that flatters this evaluation and generalises to
nothing. `ms-marco-MiniLM-L-6-v2` is trained on ~500k real query/passage pairs
and costs nothing to adopt. Fine-tuning becomes honest when the corpus has
thousands of labelled queries, not sixty; until then this is the stronger
engineering choice, and `docs/results.md` reports what it actually bought.
"""

from __future__ import annotations

from dataclasses import replace

from .retrieve import Hit

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Lazily loaded so importing the package never pulls in torch."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.name = model_name

    def rerank(self, question: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(question, h.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        # Keep the fused score visible for tracing; `score` becomes the reranker's,
        # so downstream consumers compare like with like.
        return [replace(hits[i], score=float(scores[i])) for i in order[:k]]
