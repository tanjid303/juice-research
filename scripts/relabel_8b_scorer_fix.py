"""Relabel 8B detail JSONL with fixed scorer; recompute MR★ / AME / transitions."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\tt")
sys.path.insert(0, str(ROOT))

from common.scoring import label_conflict_response  # noqa: E402
from common.stats_analysis import analyze_behavioral  # noqa: E402

DETAIL = Path(r"C:\Users\Tanjid\Downloads\mini_pilot_8b_outputs\detail_20260824T180723Z.jsonl")
DATA = Path(r"D:\tt\data\conflictqa_pilot250_popular.jsonl")
OUT_DETAIL = Path(r"C:\Users\Tanjid\Downloads\mini_pilot_8b_outputs\detail_relabeled_scorer_fix.jsonl")
OUT_REPORT = Path(r"C:\Users\Tanjid\Downloads\mini_pilot_8b_outputs\scorer_fix_report.txt")

LABELS = ["parametric", "context", "rejects_context", "other"]
BARE_CHECK = [
    "ConflictQA-popQA-chatgpt_7673",  # Kareličy
    "ConflictQA-popQA-chatgpt_7756",  # Hamilton
    "ConflictQA-popQA-chatgpt_7913",  # Iconium
]
# Previously false-context rejection examples from confound analysis
REJECTION_CHECK = [
    "ConflictQA-popQA-chatgpt_4185",
    "ConflictQA-popQA-chatgpt_4382",
    "ConflictQA-popQA-chatgpt_6091",
    "ConflictQA-popQA-chatgpt_7675",
    "ConflictQA-popQA-chatgpt_7685",
    "ConflictQA-popQA-chatgpt_7749",
    "ConflictQA-popQA-chatgpt_7779",
    "ConflictQA-popQA-chatgpt_7809",
    "ConflictQA-popQA-chatgpt_7907",
    "ConflictQA-popQA-chatgpt_7934",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    items = {
        json.loads(line)["id"]: json.loads(line)
        for line in DATA.open(encoding="utf-8")
        if line.strip()
    }
    rows = [json.loads(line) for line in DETAIL.open(encoding="utf-8") if line.strip()]

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    # Relabel
    n_changed = 0
    changes: list[tuple[str, str, str, str, str]] = []
    for r in rows:
        it = items[r["id"]]
        old = r["conflict_label"]
        new = label_conflict_response(
            r["context_conflicts"],
            it["parametric_answer"],
            it["context_answer"],
            question=it["question"],
        )
        r["conflict_label_old"] = old
        r["conflict_label"] = new
        if old != new:
            n_changed += 1
            changes.append((r["id"], r["precision"], old, new, r["context_conflicts"]))

    with OUT_DETAIL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out(f"Relabeled {len(rows)} rows; {n_changed} labels changed.")
    out(f"Wrote {OUT_DETAIL}")
    out()

    # Spot-check bare names + rejection cases (INT4)
    by_key = {(r["id"], r["precision"]): r for r in rows}
    out("=== Spot-check: bare context names (expect context) ===")
    for i in BARE_CHECK:
        r = by_key[(i, "int4")]
        it = items[i]
        out(f"  {i}: {r['conflict_label_old']} -> {r['conflict_label']}")
        out(f"    INT4: {r['context_conflicts']!r}")
        out(f"    context gold: {it['context_answer']}")
    out()
    out("=== Spot-check: prior false context rejections (INT4) ===")
    for i in REJECTION_CHECK:
        r = by_key[(i, "int4")]
        out(f"  {i}: {r['conflict_label_old']} -> {r['conflict_label']}")
        out(f"    INT4: {r['context_conflicts'][:160].replace(chr(10), ' ')}")
    out()

    # Build dataframe for stats
    df = pd.DataFrame(rows)
    shared_fp = set(df.loc[(df.precision == "fp16") & df.closed_book_correct, "id"])
    shared_i4 = set(df.loc[(df.precision == "int4") & df.closed_book_correct, "id"])
    shared = sorted(shared_fp & shared_i4)
    out(f"Shared MR* n = {len(shared)}")
    out()

    fp16 = {r["id"]: r for r in rows if r["precision"] == "fp16"}
    int4 = {r["id"]: r for r in rows if r["precision"] == "int4"}

    # Unfiltered MR (all 250)
    out("=== Unfiltered MR (all items per precision) ===")
    for prec, d in [("FP16", fp16), ("INT4", int4)]:
        labs = [d[i]["conflict_label"] for i in sorted(d)]
        mr = sum(1 for x in labs if x == "parametric") / len(labs)
        out(f"  {prec}: MR={mr:.3f} ({sum(1 for x in labs if x == 'parametric')}/{len(labs)})")
    out()

    # Label breakdown shared
    out("=== Label breakdown (shared MR*, corrected) ===")
    for name, d in [("FP16", fp16), ("INT4", int4)]:
        c = Counter(d[i]["conflict_label"] for i in shared)
        parts = ", ".join(f"{k}={c.get(k, 0)}" for k in LABELS)
        out(f"  {name}: {parts}  (total={sum(c.values())})")
    fp_c = Counter(fp16[i]["conflict_label"] for i in shared)
    i4_c = Counter(int4[i]["conflict_label"] for i in shared)
    out("  Deltas (INT4 - FP16):")
    for k in LABELS:
        out(f"    {k}: {i4_c.get(k, 0) - fp_c.get(k, 0):+d}")
    out()

    # Transition table
    out("=== Transition table FP16 -> INT4 (corrected) ===")
    trans = Counter(
        (fp16[i]["conflict_label"], int4[i]["conflict_label"]) for i in shared
    )
    for a in LABELS:
        for b in LABELS:
            n = trans.get((a, b), 0)
            if n or (a in ("parametric", "context", "other") and b in ("parametric", "context", "other")):
                out(f"  {a} -> {b}: {n}")
    out()

    # Compare old vs new context counts on shared INT4
    out("=== Old vs new on shared set ===")
    old_fp = Counter(fp16[i]["conflict_label_old"] for i in shared)
    old_i4 = Counter(int4[i]["conflict_label_old"] for i in shared)
    out(f"  FP16 old: {dict(old_fp)}")
    out(f"  FP16 new: {dict(fp_c)}")
    out(f"  INT4 old: {dict(old_i4)}")
    out(f"  INT4 new: {dict(i4_c)}")
    out(
        f"  INT4 context: {old_i4.get('context', 0)} -> {i4_c.get('context', 0)} "
        f"(delta {i4_c.get('context', 0) - old_i4.get('context', 0):+d})"
    )
    out(
        f"  FP16 context: {old_fp.get('context', 0)} -> {fp_c.get('context', 0)} "
        f"(delta {i4_c.get('context', 0) - old_fp.get('context', 0):+d} vs old FP16 baseline for AME direction)"
    )
    out()

    # Full pipeline stats
    stats = analyze_behavioral(df, n_boot=1000, seed=42)
    boot = stats["bootstrap_mr_star"]
    ame = stats["logistic_mixed_ame"]

    out("=== Bootstrap MR* (corrected) ===")
    out(f"  n_shared={boot['n_shared']}")
    d = boot["delta_int4_minus_fp16"]
    out(
        f"  FP16 MR*: {boot['fp16_mr_star']['point']:.3f} "
        f"CI95=[{boot['fp16_mr_star']['ci95'][0]:.3f}, {boot['fp16_mr_star']['ci95'][1]:.3f}]"
    )
    out(
        f"  INT4 MR*: {boot['int4_mr_star']['point']:.3f} "
        f"CI95=[{boot['int4_mr_star']['ci95'][0]:.3f}, {boot['int4_mr_star']['ci95'][1]:.3f}]"
    )
    out(
        f"  Delta INT4-FP16: {d['point_pp']:+.1f} pp "
        f"CI95=[{d['ci95_pp'][0]:+.1f}, {d['ci95_pp'][1]:+.1f}] "
        f"excludes_0={d['excludes_zero']}"
    )
    out()

    out("=== follows_context AME (corrected) ===")
    out(f"  method={ame.get('method')}")
    out(
        f"  AME: {ame.get('ame_pp', float('nan')):+.1f} pp "
        f"CI95=[{ame['ame_ci95_pp'][0]:+.1f}, {ame['ame_ci95_pp'][1]:+.1f}] "
        f"excludes_0={ame.get('ci_excludes_zero')}"
    )
    if "ame_item_averaged_pp" in ame:
        out(f"  AME item-averaged: {ame['ame_item_averaged_pp']:+.1f} pp")
    if "cluster_robust_logit" in ame:
        cr = ame["cluster_robust_logit"]
        out(
            f"  cluster-robust logit AME: {cr['ame_pp']:+.1f} pp "
            f"CI95=[{cr['ame_ci95_pp'][0]:+.1f}, {cr['ame_ci95_pp'][1]:+.1f}] "
            f"excludes_0={cr['ci_excludes_zero']}"
        )
    out()

    # Raw context rates for AME sanity
    fp_ctx = sum(1 for i in shared if fp16[i]["conflict_label"] == "context") / len(shared)
    i4_ctx = sum(1 for i in shared if int4[i]["conflict_label"] == "context") / len(shared)
    out(f"  Raw P(context) FP16={fp_ctx:.3f}, INT4={i4_ctx:.3f}, delta={(i4_ctx - fp_ctx) * 100:+.1f} pp")
    out()

    # List all INT4 label changes on shared set
    out("=== Shared-set INT4 label changes ===")
    for i in shared:
        o, n = int4[i]["conflict_label_old"], int4[i]["conflict_label"]
        if o != n:
            out(f"  {i}: {o} -> {n} | {int4[i]['context_conflicts'][:120].replace(chr(10), ' ')}")
    out()
    out("=== Shared-set FP16 label changes ===")
    for i in shared:
        o, n = fp16[i]["conflict_label_old"], fp16[i]["conflict_label"]
        if o != n:
            out(f"  {i}: {o} -> {n} | {fp16[i]['context_conflicts'][:120].replace(chr(10), ' ')}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
