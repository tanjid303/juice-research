"""8B FP16 vs INT4 confound checks on shared MR* set (same as 1B analysis)."""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DETAIL = Path(r"C:\Users\Tanjid\Downloads\mini_pilot_8b_outputs\detail_20260824T180723Z.jsonl")
DATA = Path(r"D:\tt\data\conflictqa_pilot250_popular.jsonl")
OUT = Path(r"C:\Users\Tanjid\Downloads\mini_pilot_8b_outputs\confound_report.txt")

LABELS = ["parametric", "context", "other"]


def main() -> None:
    items = {
        json.loads(line)["id"]: json.loads(line)
        for line in DATA.open(encoding="utf-8")
        if line.strip()
    }

    rows = [json.loads(line) for line in DETAIL.open(encoding="utf-8") if line.strip()]
    fp16 = {r["id"]: r for r in rows if r["precision"] == "fp16"}
    int4 = {r["id"]: r for r in rows if r["precision"] == "int4"}
    shared = sorted(set(fp16) & set(int4))
    shared_mr = [
        i
        for i in shared
        if fp16[i]["closed_book_correct"] and int4[i]["closed_book_correct"]
    ]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)

    def ntok(s: str) -> int:
        return len(tok.encode(s or "", add_special_tokens=False))

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out(f"Shared MR* n = {len(shared_mr)}")
    out()

    # 1. Label breakdown
    out("=== 1. Label breakdown (shared MR*) ===")
    for name, d in [("FP16", fp16), ("INT4", int4)]:
        c = Counter(d[i]["conflict_label"] for i in shared_mr)
        parts = ", ".join(f"{k}={c.get(k, 0)}" for k in LABELS)
        out(f"{name}: {parts}  (total={sum(c.values())})")
    fp_c = Counter(fp16[i]["conflict_label"] for i in shared_mr)
    i4_c = Counter(int4[i]["conflict_label"] for i in shared_mr)
    out("Deltas (INT4 - FP16):")
    for k in LABELS:
        out(f"  {k}: {i4_c.get(k, 0) - fp_c.get(k, 0):+d}")
    out()

    # 2. Transition table
    out("=== 2. Full transition table FP16 -> INT4 ===")
    trans = Counter(
        (fp16[i]["conflict_label"], int4[i]["conflict_label"]) for i in shared_mr
    )
    for a in LABELS:
        for b in LABELS:
            out(f"  {a} -> {b}: {trans.get((a, b), 0)}")
    out()

    p2c = [
        i
        for i in shared_mr
        if fp16[i]["conflict_label"] == "parametric"
        and int4[i]["conflict_label"] == "context"
    ]
    p2o = [
        i
        for i in shared_mr
        if fp16[i]["conflict_label"] == "parametric"
        and int4[i]["conflict_label"] == "other"
    ]
    p2p = [
        i
        for i in shared_mr
        if fp16[i]["conflict_label"] == "parametric"
        and int4[i]["conflict_label"] == "parametric"
    ]

    def len_stats(ids: list[str], title: str) -> None:
        if not ids:
            out(f"{title}: n=0")
            return
        fp_lens = [ntok(fp16[i]["context_conflicts"]) for i in ids]
        i4_lens = [ntok(int4[i]["context_conflicts"]) for i in ids]
        diffs = [i4 - fp for fp, i4 in zip(fp_lens, i4_lens)]

        def summ(x: list[int]) -> str:
            return (
                f"n={len(x)}, mean={st.mean(x):.2f}, median={st.median(x):.1f}, "
                f"min={min(x)}, max={max(x)}"
            )

        out(f"=== 3. Token length — {title} (n={len(ids)}) ===")
        out(f"  FP16 conflict output: {summ(fp_lens)}")
        out(f"  INT4 conflict output: {summ(i4_lens)}")
        out(f"  INT4-FP16 delta: mean={st.mean(diffs):+.2f}, median={st.median(diffs):+.1f}")
        out()

    # 3. Token lengths
    out("=== 3. Token lengths (Llama-3.1 tokenizer, no special tokens, context_conflicts field) ===")
    len_stats(p2c, "parametric -> context")
    len_stats(p2o, "parametric -> other")
    len_stats(p2p, "parametric -> parametric (stable)")
    fp_all = [ntok(fp16[i]["context_conflicts"]) for i in shared_mr]
    i4_all = [ntok(int4[i]["context_conflicts"]) for i in shared_mr]
    out(f"ALL shared MR* conflict outputs:")
    out(f"  FP16: n={len(fp_all)}, mean={st.mean(fp_all):.2f}, median={st.median(fp_all):.1f}")
    out(f"  INT4: n={len(i4_all)}, mean={st.mean(i4_all):.2f}, median={st.median(i4_all):.1f}")
    out()

    # Per-item lengths for p2c and p2o
    out("=== 3b. Per-item token counts (parametric->context) ===")
    for i in p2c:
        fp_t = ntok(fp16[i]["context_conflicts"])
        i4_t = ntok(int4[i]["context_conflicts"])
        out(f"  {i}: FP16={fp_t}, INT4={i4_t}, delta={i4_t - fp_t:+d}")
    out()

    out("=== 3c. Per-item token counts (parametric->other) ===")
    for i in p2o:
        fp_t = ntok(fp16[i]["context_conflicts"])
        i4_t = ntok(int4[i]["context_conflicts"])
        out(f"  {i}: FP16={fp_t}, INT4={i4_t}, delta={i4_t - fp_t:+d}")
    out()

    # 4. Raw examples parametric -> context (all if <=15, else first 15)
    out("=== 4. Raw examples: parametric -> context ===")
    out(f"Total parametric->context flips: {len(p2c)}")
    out()
    show_p2c = p2c  # user asked for 10-15; show all 26 with full text
    for idx, i in enumerate(show_p2c, start=1):
        it = items[i]
        out(f"--- Example {idx}/{len(show_p2c)} | id={i} ---")
        out(f"QUESTION: {it['question']}")
        out(f"PARAMETRIC GOLD: {it['parametric_answer']}")
        out(f"CONTEXT GOLD: {it['context_answer']}")
        out(f"FP16 OUTPUT: {fp16[i]['context_conflicts']}")
        out(f"INT4 OUTPUT: {int4[i]['context_conflicts']}")
        out(f"FP16 tokens: {ntok(fp16[i]['context_conflicts'])}, INT4 tokens: {ntok(int4[i]['context_conflicts'])}")
        out()

    # 5. parametric -> other full dump
    out("=== 5. Raw examples: parametric -> other ===")
    out(f"Total parametric->other flips: {len(p2o)}")
    out()
    for idx, i in enumerate(p2o, start=1):
        it = items[i]
        fp_t = ntok(fp16[i]["context_conflicts"])
        i4_t = ntok(int4[i]["context_conflicts"])
        out(f"--- Example {idx}/{len(p2o)} | id={i} | FP16 tok={fp_t}, INT4 tok={i4_t} ---")
        out(f"QUESTION: {it['question']}")
        out(f"PARAMETRIC GOLD: {it['parametric_answer']}")
        out(f"CONTEXT GOLD: {it['context_answer']}")
        out(f"FP16 OUTPUT: {fp16[i]['context_conflicts']}")
        out(f"INT4 OUTPUT: {int4[i]['context_conflicts']}")
        out()

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
