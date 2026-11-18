"""ConflictQA prompt templates (instruct chat template)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Condition = Literal["closed_book", "context_agrees", "context_conflicts"]
PromptType = Literal["clean", "substitution_conflict", "coherent_conflict"]


@dataclass
class ConflictItem:
    id: str
    question: str
    parametric_answer: str
    context_answer: str
    passage_agrees: str
    passage_conflicts: str
    source: str = "ConflictQA"


def build_user_content(item: ConflictItem, condition: Condition) -> str:
    if condition == "closed_book":
        return item.question.strip()
    if condition == "context_agrees":
        return (
            f"Context: {item.passage_agrees.strip()}\n\n"
            f"Question: {item.question.strip()}\n"
            "Answer with a short phrase only."
        )
    if condition == "context_conflicts":
        return (
            f"Context: {item.passage_conflicts.strip()}\n\n"
            f"Question: {item.question.strip()}\n"
            "Answer with a short phrase only."
        )
    raise ValueError(condition)


def apply_chat_template(tokenizer, item: ConflictItem, condition: Condition) -> str:
    user = build_user_content(item, condition)
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def juice_style_factual_prompts(item: ConflictItem) -> dict[str, str]:
    """
    Plain factual-completion prompts (no chat template), JuICE-shaped:

    - clean: elicit answer with no conflict
    - substitution_conflict: short wrong-answer prefix then re-ask (JuICE subst pattern)
    - coherent_conflict: conflicting passage then Question/Answer cloze

    Used for base-model ID diagnostics to match GaotangLi/JUICE prompt family.
    """
    q = item.question.strip()
    ctx = item.context_answer.strip()
    passage = item.passage_conflicts.strip()
    return {
        "clean": f"Question: {q}\nAnswer:",
        "substitution_conflict": (
            f"Question: {q}\nAnswer: {ctx}\nQuestion: {q}\nAnswer:"
        ),
        "coherent_conflict": f"{passage}\n\nQuestion: {q}\nAnswer:",
    }
