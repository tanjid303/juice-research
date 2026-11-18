"""Answer normalization and conflict labels (parametric / context / rejects_context / other)."""

from __future__ import annotations

import re
from typing import Literal, Sequence

Label = Literal["parametric", "context", "rejects_context", "other"]

_STOP = {
    "a",
    "an",
    "the",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "being",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "as",
    "by",
    "with",
    "from",
    "that",
    "this",
    "his",
    "her",
    "their",
    "who",
    "whom",
    "whose",
    "which",
    "what",
    "when",
    "where",
    "how",
    "note",
    "spelling",
    "occupation",
    "he",
    "she",
    "they",
    "it",
}

# Sentence-level / local cues that the mentioned entity is being denied.
_NEGATION_PHRASES = (
    " not ",
    " no ",
    " isn't ",
    " isnt ",
    " aren't ",
    " arent ",
    " wasn't ",
    " wasnt ",
    " weren't ",
    " werent ",
    " incorrect ",
    " false ",
    " mistake ",
    " wrong ",
    " no evidence ",
    " never ",
    " rather than ",
    " instead of ",
    " i can't confirm ",
    " i cant confirm ",
    " i couldn't find ",
    " i couldnt find ",
    " i could not find ",
    " i couldn't verify ",
    " i couldnt verify ",
    " i can't verify ",
    " i cant verify ",
    " i can't answer ",
    " i cant answer ",
    " i can't provide ",
    " i cant provide ",
    " i need to correct ",
    " i m afraid ",
    " i'm afraid ",
    " not the capital ",
    " not the producer ",
    " not the director ",
    " not the composer ",
    " not the screenwriter ",
    " not involved ",
    " does not ",
    " did not ",
    " doesn't ",
    " doesnt ",
    " didn't ",
    " didnt ",
)


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"^(answer:|the answer is|answers?:)\s*", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_tokens(text: str) -> list[str]:
    return [t for t in normalize_answer(text).split() if t not in _STOP and len(t) > 1]


def is_refusal(pred: str) -> bool:
    p = normalize_answer(pred)
    cues = (
        "i don t have",
        "i do not have",
        "i couldn t find",
        "i could not find",
        "no information",
        "not sure",
        "i m not sure",
        "unknown",
        "cannot find",
        "can t find",
        "i don t know",
        "i do not know",
        "could you provide more context",
        "is there anything else i can help",
    )
    return any(c in p for c in cues)


def _extract_role_aliases(ref: str) -> list[str]:
    """Pull short occupation/entity phrases from long memory_answer sentences."""
    aliases: list[str] = []
    n = normalize_answer(ref)
    for m in re.finditer(
        r"\b(?:is|was|are|were)\s+(?:an?\s+|the\s+)?([a-z0-9 ]{2,60}?)(?:\.|$)",
        n,
    ):
        phrase = m.group(1).strip()
        if phrase and phrase not in _STOP:
            aliases.append(phrase)
    for m in re.finditer(r"occupation is\s+([a-z0-9 ]{2,40})", n):
        aliases.append(m.group(1).strip())
    # Leading subject entity: "Hamilton is the capital of New Zealand."
    parts = re.split(r"\s+(?:is|was|are|were)\s+", n, maxsplit=1)
    if len(parts) == 2 and parts[0].strip():
        subj = parts[0].strip()
        if 1 <= len(content_tokens(subj)) <= 6:
            aliases.append(subj)
    return aliases


def _answer_tokens(ref: str, question: str | None = None) -> list[str]:
    """Distinctive answer tokens in ref after stripping question-name echoes."""
    q_toks = set(content_tokens(question or ""))
    return [t for t in content_tokens(ref) if t not in q_toks]


def _alias_in_pred(pred: str, alias: str, question: str | None = None) -> bool:
    """Match a short answer alias; ignore tokens that only repeat the question name."""
    r = normalize_answer(alias)
    p = normalize_answer(pred)
    if not r or not p:
        return False
    if r in p:
        return True
    q_toks = set(content_tokens(question or ""))
    r_toks = [t for t in content_tokens(alias) if t not in q_toks]
    if not r_toks:
        return False
    p_toks = set(content_tokens(pred))
    return all(t in p_toks for t in r_toks)


def _mentions_answer(pred: str, ref: str, *, question: str | None = None) -> bool:
    """True if pred mentions the distinctive entity/answer tokens from ref."""
    toks = _answer_tokens(ref, question)
    if not toks:
        return False
    p_toks = set(content_tokens(pred))
    # Single-entity answers (most ConflictQA capitals/names): require the entity token(s).
    if len(toks) <= 2:
        return all(t in p_toks for t in toks)
    # Longer answers: majority of distinctive tokens.
    hit = sum(1 for t in toks if t in p_toks)
    return hit >= max(2, int(0.6 * len(toks) + 0.999))


def _padded(text: str) -> str:
    return f" {normalize_answer(text)} "


def _has_negation_cue(text: str) -> bool:
    padded = _padded(text)
    return any(c in padded for c in _NEGATION_PHRASES)


def _entity_positively_asserted(pred: str, ent_toks: list[str]) -> bool:
    """True if pred clearly asserts the entity as the answer (not merely mentions it)."""
    if not ent_toks:
        return False
    p = normalize_answer(pred)
    ent = " ".join(ent_toks)
    esc = re.escape(ent)
    p_toks = content_tokens(pred)

    # Drop denial-that clauses so "can't verify that X is the capital" does not
    # count as asserting X.
    p = re.sub(
        rf"(?:can\s*t|cannot|couldn\s*t|could\s+not)\s+"
        rf"(?:verify|confirm|find|provide).{{0,60}}"
        rf"(?:that\s+)?{esc}\s+(?:is|was|are|were|being).{{0,80}}",
        " ",
        p,
    )
    p = re.sub(
        rf"(?:can\s*t|cannot|couldn\s*t|could\s+not)\s+find.{{0,80}}"
        rf"confirming\s+(?:that\s+)?{esc}.{{0,60}}",
        " ",
        p,
    )
    p = re.sub(r"\s+", " ", p).strip()

    # Bare / near-bare answer that is just the entity.
    if len(p_toks) <= max(3, len(ent_toks) + 1) and all(t in set(p.split()) or t in p_toks for t in ent_toks):
        # Recompute on original for bare answers (denial strip may empty the string).
        if len(content_tokens(pred)) <= max(3, len(ent_toks) + 1) and all(
            t in content_tokens(pred) for t in ent_toks
        ):
            if not _entity_denied(pred, ent_toks):
                return True

    # "the (actual/correct) capital/answer … is {entity}"
    if re.search(
        rf"\b(?:actual|correct|real)\s+"
        rf"(?:capital|answer|producer|director|composer|screenwriter|author|mother|occupation)"
        rf".{{0,40}}\b(?:is|was)\s+(?:actually\s+|really\s+)?{esc}\b",
        p,
    ):
        return True
    if re.search(
        rf"\b(?:capital|answer|producer|director|composer|screenwriter|author)"
        rf".{{0,40}}\b(?:is|was)\s+(?:actually\s+|really\s+)?{esc}\b",
        p,
    ):
        return True
    # "{entity} is the capital/producer/…"
    if re.search(
        rf"\b{esc}\s+(?:is|was)\s+(?:an?\s+|the\s+)?"
        rf"(?:capital|producer|director|composer|screenwriter|author|mother)\b",
        p,
    ):
        return True
    # "capital of {entity}" when not negated ("not the capital of {entity}")
    if re.search(rf"\bcapital\s+of\s+{esc}\b", p) and not re.search(
        rf"\b(?:not|never)\s+(?:the\s+)?capital\s+of\s+{esc}\b", p
    ):
        return True
    # "correct answer is {entity}"
    if re.search(rf"\bcorrect answer\s+(?:is|was)\s+(?:actually\s+|really\s+)?{esc}\b", p):
        return True
    return False


def _entity_denied(pred: str, ent_toks: list[str]) -> bool:
    """
    True if the distinctive entity tokens are locally denied / rejected.
    Requires negation bound to the entity (not merely 'not'/'incorrect' elsewhere).
    """
    if not ent_toks:
        return False
    p = normalize_answer(pred)
    ent = " ".join(ent_toks)
    esc = re.escape(ent)

    # "{entity} is/was not …"
    if re.search(rf"\b{esc}\s+(?:is|was|are|were)\s+not\b", p):
        return True
    if re.search(rf"\b{esc}\s+isn\s*t\b", p):
        return True
    # "… is/was not … {entity}" (e.g. "is not the capital of Kangra")
    if re.search(rf"\b(?:is|was|are|were)\s+not\b.{{0,80}}\b{esc}\b", p):
        return True
    # bare "not {entity}" (e.g. "Indianapolis, not Vilne" denies Vilne)
    if re.search(rf"\bnot\s+{esc}\b", p):
        return True
    # "Rumbek is not a city in Wisconsin" style
    if re.search(rf"\b{esc}\s+(?:is|was)\s+not\s+a\b", p):
        return True
    # confirmation hedges that bind the entity as the denied claim
    if re.search(
        rf"(?:can\s*t|cannot|couldn\s*t|could\s+not)\s+"
        rf"(?:find|verify|confirm).{{0,80}}"
        rf"(?:confirming|confirms|that|about)\s+{esc}\b",
        p,
    ):
        return True
    if re.search(
        rf"(?:can\s*t|cannot|couldn\s*t|could\s+not)\s+"
        rf"(?:verify|confirm)\b.{{0,100}}\b{esc}\b",
        p,
    ):
        return True
    if re.search(
        rf"(?:can\s*t|cannot|couldn\s*t|could\s+not)\s+find.{{0,100}}\b{esc}\b",
        p,
    ):
        return True
    if re.search(rf"\bno evidence\b.{{0,80}}\b{esc}\b", p):
        return True
    if re.search(rf"\bno information\b.{{0,80}}\b{esc}\b", p):
        return True
    if re.search(rf"\bnot aware\b.{{0,100}}\b{esc}\b", p):
        return True
    # "{entity} … incorrect/false" — entity is the rejected claim
    if re.search(rf"\b{esc}\b.{{0,40}}\b(?:incorrect|false|wrong)\b", p):
        return True
    if re.search(rf"\bmistake\b.{{0,80}}\b{esc}\s+(?:is|was|are|were)\s+not\b", p):
        return True
    if re.search(rf"\bneed to correct\b.{{0,100}}\b{esc}\s+(?:is|was|are|were)\s+not\b", p):
        return True
    if re.search(rf"\bconfirming\b.{{0,40}}\b{esc}\b", p) and _has_negation_cue(pred):
        return True
    # "I can't verify that {entity} is …"
    if re.search(
        rf"(?:can\s*t|cannot|couldn\s*t|could\s+not)\s+"
        rf"(?:verify|confirm|find).{{0,40}}\bthat\s+{esc}\s+(?:is|was|are|were)\b",
        p,
    ):
        return True
    return False


def is_rejecting_mention(
    pred: str,
    ref: str,
    *,
    question: str | None = None,
) -> bool:
    """
    True if pred mentions ref's entity/answer but rejects or denies it.
    A clear positive assertion of the entity overrides denial hedges aimed at
    a different (context) claim in the same string.
    """
    if not pred or not ref:
        return False
    ent = _answer_tokens(ref, question)
    if not ent:
        return False
    if not _mentions_answer(pred, ref, question=question):
        return False
    # Check denial first on the raw string; positive assertion uses denial-clause stripping.
    denied = _entity_denied(pred, ent)
    if not denied:
        return False
    if _entity_positively_asserted(pred, ent):
        return False
    return True


def answers_match(
    pred: str,
    ref: str,
    aliases: Sequence[str] | None = None,
    *,
    question: str | None = None,
    allow_rejected_mention: bool = False,
) -> bool:
    """
    True if pred asserts ref. By default, a mention of ref inside a negation /
    rejection frame does not count as a match.
    """
    if not pred or not ref:
        return False
    if not allow_rejected_mention and is_rejecting_mention(pred, ref, question=question):
        return False

    candidates: list[str] = []
    if aliases:
        candidates.extend([a for a in aliases if a])
    candidates.extend(_extract_role_aliases(ref))
    candidates.append(ref)

    # Prefer short role aliases first (occupation / entity), not the full sentence.
    candidates = sorted(set(candidates), key=lambda s: len(content_tokens(s)))

    for cand in candidates:
        if _alias_in_pred(pred, cand, question=question):
            # Full-sentence candidate: require answer tokens beyond the question name.
            answer_toks = _answer_tokens(cand, question)
            # Skip if the candidate contributes no distinctive tokens (question echo only).
            if not answer_toks:
                continue
            if len(content_tokens(cand)) >= 4:
                p_toks = set(content_tokens(pred))
                hit = sum(1 for t in answer_toks if t in p_toks)
                # Single-token entities (Hamilton, Vilne, …) must fully hit;
                # longer answers need majority coverage (at least 2).
                if len(answer_toks) <= 2:
                    if hit < len(answer_toks):
                        continue
                elif hit < max(2, int(0.6 * len(answer_toks) + 0.999)):
                    continue
            return True
    return False


def label_conflict_response(
    pred: str,
    parametric_answer: str,
    context_answer: str,
    *,
    parametric_aliases: Sequence[str] | None = None,
    context_aliases: Sequence[str] | None = None,
    question: str | None = None,
) -> Label:
    """
    Three-way behavioral labels plus rejects_context for denied context mentions.

    Order:
      1. Asserted parametric answer → parametric
      2. Asserted context answer (not in a negation frame) → context
      3. Context entity mentioned but rejected, without asserted parametric → rejects_context
      4. Else → other
    """
    if answers_match(pred, parametric_answer, parametric_aliases, question=question):
        return "parametric"
    if answers_match(pred, context_answer, context_aliases, question=question):
        return "context"
    # Context mentioned only inside a rejection / denial frame.
    if is_rejecting_mention(pred, context_answer, question=question):
        return "rejects_context"
    return "other"


def closed_book_correct(
    pred: str,
    gold_parametric: str,
    aliases: Sequence[str] | None = None,
    *,
    question: str | None = None,
) -> bool:
    """
    Closed-book correctness for ConflictQA-style targets.
    Pure refusals that only echo the entity name do not count.
    Mentions of the answer role inside a hedge (e.g. 'including Jim Brown, an American
    football player') can still count via role-alias match.
    """
    if not pred or not gold_parametric:
        return False
    roles = list(aliases or []) + _extract_role_aliases(gold_parametric)
    # Strong path: short role alias appears in prediction.
    for role in roles:
        if 1 <= len(content_tokens(role)) <= 6 and _alias_in_pred(pred, role, question=question):
            return True
    if is_refusal(pred):
        return False
    return answers_match(pred, gold_parametric, aliases, question=question)


def memorization_ratio(labels: list[Label]) -> float:
    if not labels:
        return float("nan")
    return sum(1 for x in labels if x == "parametric") / len(labels)
