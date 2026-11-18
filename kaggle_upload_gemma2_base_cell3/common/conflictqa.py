"""ConflictQA / ConflictBank JSONL I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from common.prompts import ConflictItem


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            "See data/README.md for download and conversion instructions."
        )
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_to_conflict_item(row: dict) -> ConflictItem:
    """Normalize our filtered JSONL schema to ConflictItem."""
    return ConflictItem(
        id=str(row.get("id", row.get("qid", "unknown"))),
        question=row["question"],
        parametric_answer=row["parametric_answer"],
        context_answer=row["context_answer"],
        passage_agrees=row["passage_agrees"],
        passage_conflicts=row["passage_conflicts"],
        source=row.get("source", "ConflictQA"),
    )


def load_conflict_items(path: str | Path, n_items: int | None = None) -> List[ConflictItem]:
    rows = load_jsonl(path)
    if n_items is not None:
        rows = rows[:n_items]
    return [row_to_conflict_item(r) for r in rows]


def convert_hf_conflictqa_row(row: dict, idx: int, config: str) -> dict:
    """
    Convert osunlp/ConflictQA popQA fields to our JSONL schema.

    ConflictQA-popQA-* keys (verified):
      question, ground_truth (list), memory_answer, parametric_memory,
      counter_answer, counter_memory,
      parametric_memory_aligned_evidence, counter_memory_aligned_evidence
    """
    question = row.get("question") or row.get("query") or ""
    ground = row.get("ground_truth") or []
    if isinstance(ground, str):
        ground = [ground]

    # Parametric side of the conflict pair = ChatGPT parametric memory answer.
    # Do NOT prefer Wikidata ground_truth here: for many rows it equals the *counter*
    # answer (e.g. occupation=politician), which would collapse the three-way labels.
    parametric = (
        row.get("parametric_answer")
        or row.get("memory_answer")
        or (ground[0] if ground else "")
        or row.get("answer")
        or ""
    )
    context_ans = (
        row.get("context_answer")
        or row.get("counter_answer")
        or row.get("memory_answer")
        or ""
    )
    passage_agrees = (
        row.get("passage_agrees")
        or row.get("parametric_memory_aligned_evidence")
        or row.get("parametric_memory")
        or row.get("context")
        or ""
    )
    passage_conflicts = (
        row.get("passage_conflicts")
        or row.get("counter_memory_aligned_evidence")
        or row.get("counter_memory")
        or row.get("counter_context")
        or ""
    )
    return {
        "id": str(row.get("id", f"{config}_{idx}")),
        "question": question,
        "parametric_answer": parametric,
        "context_answer": context_ans,
        "passage_agrees": passage_agrees,
        "passage_conflicts": passage_conflicts,
        "ground_truth": ground,
        "memory_answer": row.get("memory_answer", ""),
        "popularity": row.get("popularity"),
        "source": config,
    }


def export_conflictqa_from_hf(
    output_path: str | Path,
    config: str = "ConflictQA-popQA-chatgpt",
    split: str = "test",
    max_items: int | None = None,
    *,
    min_popularity: int | None = None,
    top_by_popularity: bool = False,
    dedupe_questions: bool = True,
) -> int:
    """
    Pull ConflictQA from Hugging Face and write our JSONL format.

    top_by_popularity: keep the highest-popularity rows (denser entities for 1B CB).
    min_popularity: drop rows below this PopQA popularity score.
    """
    from datasets import load_dataset

    ds = load_dataset("osunlp/ConflictQA", config, split=split, trust_remote_code=True)
    indexed = [(i, dict(row)) for i, row in enumerate(ds)]
    if min_popularity is not None:
        indexed = [(i, r) for i, r in indexed if int(r.get("popularity") or 0) >= min_popularity]
    if top_by_popularity:
        indexed.sort(key=lambda x: int(x[1].get("popularity") or 0), reverse=True)

    rows = []
    seen_q: set[str] = set()
    for i, row in indexed:
        q = (row.get("question") or "").strip().lower()
        if dedupe_questions and q in seen_q:
            continue
        if dedupe_questions and q:
            seen_q.add(q)
        rows.append(convert_hf_conflictqa_row(row, i, config))
        if max_items is not None and len(rows) >= max_items:
            break
    save_jsonl(output_path, rows)
    return len(rows)
