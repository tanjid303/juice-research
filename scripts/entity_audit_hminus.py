"""Entity-level paired audit for Gemma identified H− and random-H− controls.

Same format as entity_audit_k2_k10.txt. Does not rewrite run JSONs or main.tex.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\tt")
sys.path.insert(0, str(ROOT))
from common.scoring import (  # noqa: E402
    label_conflict_response,
    _mentions_answer,
)

GOLD_PATH = ROOT / "data" / "conflictqa_pilot250_popular.jsonl"
OUT = ROOT / "stage2_question_aware_rescore" / "entity_audit_hminus.txt"

PAIRS = [
    # 1. n=100 identified (load-bearing)
    (
        "1a Gemma n100 FP16 identified H−",
        ROOT / "last-one-g" / "gemma_n100_fp16_hminus" / "steer_baseline.json",
        ROOT / "last-one-g" / "gemma_n100_fp16_hminus" / "suppress_context_k10.json",
    ),
    (
        "1b Gemma n100 INT4 identified H−",
        ROOT / "last-one-g" / "gemma_n100_int4_hminus" / "steer_baseline.json",
        ROOT / "last-one-g" / "gemma_n100_int4_hminus" / "suppress_context_k10.json",
    ),
    # 2. n=100 random controls
    (
        "2a Gemma n100 FP16 random H−",
        ROOT / "last-one-g" / "gemma_n100_fp16_random_hminus" / "steer_baseline.json",
        ROOT / "last-one-g" / "gemma_n100_fp16_random_hminus" / "random_suppress_k10.json",
    ),
    (
        "2b Gemma n100 INT4 random H−",
        ROOT / "last-one-g" / "gemma_n100_int4_hminus" / "steer_baseline.json",
        ROOT / "last-one-g" / "gemma_n100_int4_hminus" / "random_suppress_k10.json",
    ),
    # 3. n=50 identified
    (
        "3a Gemma n50 FP16 identified H−",
        ROOT / "task-2-res" / "juice_stage2_gemma2_fp16_context_suppress" / "steer_baseline.json",
        ROOT / "task-2-res" / "juice_stage2_gemma2_fp16_context_suppress" / "suppress_context_k10.json",
    ),
    (
        "3b Gemma n50 INT4 identified H−",
        ROOT / "gemma-int4-kicu" / "juice_stage2_gemma2_int4_context_suppress" / "steer_baseline.json",
        ROOT / "gemma-int4-kicu" / "juice_stage2_gemma2_int4_context_suppress" / "suppress_context_k10.json",
    ),
    # 4. n=50 random
    (
        "4a Gemma n50 FP16 random H−",
        ROOT / "res-fp16-test-gemma" / "juice_stage2_gemma2_fp16_random_k10_suppress" / "steer_baseline.json",
        ROOT / "res-fp16-test-gemma" / "juice_stage2_gemma2_fp16_random_k10_suppress" / "random_suppress_k10.json",
    ),
    (
        "4b Gemma n50 INT4 random H−",
        ROOT / "gemma-int4-kicu" / "juice_stage2_gemma2_int4_context_suppress" / "steer_baseline.json",
        ROOT / "gemma-int4-kicu" / "juice_stage2_gemma2_int4_context_suppress" / "random_suppress_k10.json",
    ),
]


def load_gold() -> dict:
    out = {}
    for line in GOLD_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            out[o["id"]] = o
    return out


def load_rows(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def qa_lab(text: str, g: dict) -> str:
    return label_conflict_response(
        text, g["parametric_answer"], g["context_answer"], question=g["question"]
    )


def clip(text: str, n: int = 140) -> str:
    t = " / ".join(line.strip() for line in text.splitlines() if line.strip())
    return t if len(t) <= n else t[: n - 1] + "…"


def ents(text: str, g: dict) -> str:
    ctx = _mentions_answer(text, g["context_answer"], question=g["question"])
    par = _mentions_answer(text, g["parametric_answer"], question=g["question"])
    if ctx and par:
        return "BOTH"
    if ctx:
        return "CTX"
    if par:
        return "PAR"
    return "NEITHER"


def classify_pair(b_ent: str, s_ent: str) -> str:
    if b_ent == s_ent:
        if b_ent == "CTX":
            return "same_ctx_wrap_or_rephrase"
        if b_ent == "PAR":
            return "same_par"
        if b_ent == "BOTH":
            return "same_both"
        return "same_neither"
    if b_ent in {"CTX", "BOTH", "NEITHER"} and s_ent == "PAR":
        return "toward_parametric_entity"
    if b_ent in {"PAR", "BOTH", "NEITHER"} and s_ent == "CTX":
        return "toward_context_entity"
    if b_ent == "CTX" and s_ent == "BOTH":
        return "ctx_plus_par_added"
    if b_ent == "PAR" and s_ent == "BOTH":
        return "par_plus_ctx_added"
    if b_ent == "BOTH" and s_ent == "CTX":
        return "dropped_par_kept_ctx"
    if b_ent == "BOTH" and s_ent == "PAR":
        return "dropped_ctx_kept_par"
    return f"{b_ent}->{s_ent}"


GENUINE = {
    "toward_parametric_entity",
    "toward_context_entity",
    "dropped_par_kept_ctx",
    "dropped_ctx_kept_par",
    "ctx_plus_par_added",
    "par_plus_ctx_added",
}


def analyze(name: str, bpath: Path, spath: Path, gold: dict, lines: list[str]) -> None:
    def w(s: str = "") -> None:
        lines.append(s)

    base, ste = load_rows(bpath), load_rows(spath)
    ids = sorted(set(base) & set(ste))
    w("=" * 110)
    w(name)
    w(f"base={bpath}")
    w(f"steered={spath}")
    w(f"n={len(ids)}  text_identical={sum(1 for i in ids if base[i]['text']==ste[i]['text'])}")

    rows = []
    for i in ids:
        g = gold[i]
        bt, st = base[i]["text"], ste[i]["text"]
        olb, ols = base[i]["label"], ste[i]["label"]
        nlb, nls = qa_lab(bt, g), qa_lab(st, g)
        be, se = ents(bt, g), ents(st, g)
        rows.append(
            {
                "id": i,
                "q": g["question"],
                "ctx": g["context_answer"],
                "par": g["parametric_answer"],
                "bt": bt,
                "st": st,
                "olb": olb,
                "ols": ols,
                "nlb": nlb,
                "nls": nls,
                "be": be,
                "se": se,
                "cls": classify_pair(be, se),
                "orig_fc_flip": (olb == "context") != (ols == "context"),
                "qa_fc_flip": (nlb == "context") != (nls == "context"),
                "orig_d": int(ols == "context") - int(olb == "context"),
                "qa_d": int(nls == "context") - int(nlb == "context"),
                "relabel_b": olb != nlb,
                "relabel_s": ols != nls,
            }
        )

    orig_pp = 100 * sum(r["orig_d"] for r in rows) / len(rows)
    qa_pp = 100 * sum(r["qa_d"] for r in rows) / len(rows)
    w(f"paired Δ fc orig={orig_pp:+.1f}pp  qa={qa_pp:+.1f}pp")
    w(
        f"orig fc flips={sum(r['orig_fc_flip'] for r in rows)}  "
        f"qa fc flips={sum(r['qa_fc_flip'] for r in rows)}  "
        f"baseline relabeled={sum(r['relabel_b'] for r in rows)}  "
        f"steered relabeled={sum(r['relabel_s'] for r in rows)}"
    )

    qa_flip = [r for r in rows if r["qa_fc_flip"]]
    orig_only = [r for r in rows if r["orig_fc_flip"] and not r["qa_fc_flip"]]
    new_only = [r for r in rows if r["qa_fc_flip"] and not r["orig_fc_flip"]]

    w("qa-flip class counts: " + str(dict(Counter(r["cls"] for r in qa_flip))))
    w("orig-only-flip class counts: " + str(dict(Counter(r["cls"] for r in orig_only))))

    genuine = [r for r in qa_flip if r["cls"] in GENUINE]
    wrap = [r for r in qa_flip if r["cls"] == "same_ctx_wrap_or_rephrase"]
    w(
        f"qa-flip genuine-entity-change={len(genuine)}  "
        f"same-ctx-wrap={len(wrap)}  other={len(qa_flip)-len(genuine)-len(wrap)}"
    )
    w(
        f"qa Δ from genuine flips={100*sum(r['qa_d'] for r in genuine)/len(rows):+.1f}pp  "
        f"from wrap flips={100*sum(r['qa_d'] for r in wrap)/len(rows):+.1f}pp"
    )
    to_par = [r for r in qa_flip if r["cls"] in {"toward_parametric_entity", "ctx_plus_par_added", "dropped_ctx_kept_par"}]
    to_ctx = [r for r in qa_flip if r["cls"] in {"toward_context_entity", "par_plus_ctx_added", "dropped_par_kept_ctx"}]
    leave = [r for r in qa_flip if r["cls"] == "CTX->NEITHER"]
    w(
        f"qa-flip toward gold PAR={len(to_par)}  toward gold CTX={len(to_ctx)}  "
        f"CTX->NEITHER={len(leave)}"
    )

    def dump(title: str, subset: list[dict]) -> None:
        w("")
        w(f"--- {title} (n={len(subset)}) ---")
        for r in subset:
            short = r["id"].replace("ConflictQA-popQA-chatgpt_", "")
            w(
                f"ID {short}  class={r['cls']}  ents {r['be']}→{r['se']}  "
                f"orig {r['olb']}→{r['ols']}  qa {r['nlb']}→{r['nls']}  "
                f"Δfc orig {r['orig_d']:+d} qa {r['qa_d']:+d}"
            )
            w(f"  Q: {r['q'][:100]}")
            w(f"  ctx gold: {clip(r['ctx'], 90)}")
            w(f"  par gold: {clip(r['par'], 90)}")
            w(f"  BASE [{r['olb']}→qa {r['nlb']}] {clip(r['bt'])}")
            w(f"  STE  [{r['ols']}→qa {r['nls']}] {clip(r['st'])}")
            w("")

    dump("QA paired follows_context flips (drivers of rescored Δ)", qa_flip)
    dump("ORIGINAL fc flips that VANISH after question= (L27-style wrap)", orig_only)
    dump("NEW qa fc flips that were NOT original fc flips", new_only)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    gold = load_gold()
    lines: list[str] = [
        "Gemma identified H− vs random H− — entity-level paired audit",
        "Matcher: label_conflict_response(..., question=) and _mentions_answer(..., question=)",
        "Same dump format as entity_audit_k2_k10.txt. Original run JSONs not rewritten.",
        "",
    ]
    for name, b, s in PAIRS:
        analyze(name, b, s, gold, lines)
        print("done", name)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT, "lines", len(lines))


if __name__ == "__main__":
    main()
