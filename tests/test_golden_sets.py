"""Structural contract for the golden question sets.

The deeper check — that an answerable question is actually supported by the page
it cites — lives in `scripts/verify_golden.py` and needs the corpus index, which
is gitignored. Everything here is checkable from the JSONL alone, so it runs in
CI on every push.

What these guard against, all of which have happened or nearly happened:

* a mislabelled `gold_page`, which is invisible at runtime but counts as a
  retrieval miss forever after;
* an *unanswerable* question that still carries a source, which hands the gate
  the answer it is supposed to refuse;
* duplicate ids across files, which silently drops rows when the sets are
  concatenated into one dict;
* the same question asked twice, which double-weights it in every metric.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

EVALS = Path(__file__).parent.parent / "evals"
HAND_WRITTEN = ["golden.jsonl", "golden_v2.jsonl", "golden_v3.jsonl"]


def _load(name: str) -> list[dict]:
    path = EVALS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _all_rows() -> list[tuple[str, dict]]:
    out = []
    for name in HAND_WRITTEN:
        path = EVALS / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append((name, json.loads(line)))
    return out


@pytest.mark.parametrize("name", HAND_WRITTEN)
def test_rows_have_the_required_fields(name):
    for row in _load(name):
        for key in ("id", "type", "question", "answerable", "source"):
            assert key in row, f"{row.get('id')} missing {key}"
        assert isinstance(row["answerable"], bool)
        assert row["question"].strip().endswith("?"), row["id"]


@pytest.mark.parametrize("name", HAND_WRITTEN)
def test_answerable_rows_cite_a_source(name):
    for row in _load(name):
        if not row["answerable"]:
            continue
        assert row.get("gold_doc"), f"{row['id']} answerable but no gold_doc"
        assert isinstance(row.get("gold_page"), int), (
            f"{row['id']} gold_page must be an int, got {row.get('gold_page')!r}"
        )
        assert row["gold_page"] > 0
        assert (row.get("note") or "").strip(), f"{row['id']} has no recorded answer"


@pytest.mark.parametrize("name", HAND_WRITTEN)
def test_unanswerable_rows_cite_nothing(name):
    """An unanswerable item with a gold page is a labelling bug, not a hard case."""
    for row in _load(name):
        if row["answerable"]:
            continue
        assert row.get("gold_doc") is None, f"{row['id']} unanswerable but cites a doc"
        assert row.get("gold_page") is None, (
            f"{row['id']} unanswerable but cites a page"
        )


def test_ids_are_unique_across_all_sets():
    seen: dict[str, str] = {}
    for name, row in _all_rows():
        assert row["id"] not in seen, (
            f"id {row['id']} appears in both {seen[row['id']]} and {name}"
        )
        seen[row["id"]] = name


def test_no_question_is_asked_twice():
    def norm(q: str) -> str:
        return " ".join(re.sub(r"[^a-z ]", " ", q.lower()).split())

    seen: dict[str, str] = {}
    for name, row in _all_rows():
        key = norm(row["question"])
        assert key not in seen, (
            f"{row['id']} in {name} repeats a question from {seen[key]}"
        )
        seen[key] = f"{name}:{row['id']}"


def test_unanswerable_share_stays_meaningful():
    """The gate is only exercised if a real share of questions have no answer.

    Held near a third by construction. A drift far below that would quietly make
    the abstention results look better than they are.
    """
    rows = [r for _, r in _all_rows()]
    share = sum(not r["answerable"] for r in rows) / len(rows)
    assert 0.25 <= share <= 0.45, f"unanswerable share is {share:.0%}"
