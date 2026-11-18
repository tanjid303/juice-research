"""Local diagnostics for Cell B (native INT4 K10 → INT4 N128).

Same exam-meta heuristic as scripts/cell_a_diagnostics.py.
No GPU. Writes:
  unga-bunga-23/cell_b_diagnostics.json
  unga-bunga-23/cell_b_inspection.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"d:\tt")
CELL_B = ROOT / "unga-bunga-23" / "juice_stage2_int4_ncalib128" / "steer_native_int4_context_k10.json"
INT4_BASE = ROOT / "this_is_the_128" / "steer_baseline (2).json"
FORWARD = ROOT / "this_is_the_128" / "steer_context_k10 (1).json"  # FP16 K10 → INT4 N128
FP16_K10 = ROOT / "juice_stage2_base_directional_cell3" / "steer_context_k10.json"
FP16_BASE = ROOT / "juice_stage2_base_directional_cell3" / "steer_baseline.json"
DATA = ROOT / "data" / "conflictqa_pilot250_popular.jsonl"
OUT_DIR = ROOT / "unga-bunga-23"

EXAM_META_KEYS = (
    "multiple-choice",
    "test-taker",
    "the question is",
    "note:",
    "this question is",
    "the best answer is",
    "correctness:",
)


def load_jsonl(path: Path) -> dict:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[obj["id"]] = obj
    return out


def exam_meta(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in EXAM_META_KEYS)


def label_counts(rows: dict) -> dict:
    out = {"parametric": 0, "context": 0, "rejects_context": 0, "other": 0}
    for r in rows.values():
        lab = r.get("label", "other")
        out[lab] = out.get(lab, 0) + 1
    return out


def exam_meta_ids(rows: dict) -> set[str]:
    return {i for i, r in rows.items() if exam_meta(r.get("text", ""))}


def main() -> None:
    gold = load_jsonl(DATA)
    cell_b = {r["id"]: r for r in json.loads(CELL_B.read_text(encoding="utf-8"))}
    int4_base = {r["id"]: r for r in json.loads(INT4_BASE.read_text(encoding="utf-8"))}
    forward = {r["id"]: r for r in json.loads(FORWARD.read_text(encoding="utf-8"))}
    fp16_k10 = {r["id"]: r for r in json.loads(FP16_K10.read_text(encoding="utf-8"))}
    fp16_base = {r["id"]: r for r in json.loads(FP16_BASE.read_text(encoding="utf-8"))}

    ids = list(cell_b)
    assert set(ids) == set(int4_base) == set(forward) == set(fp16_k10)

    joined = []
    for i in ids:
        g = gold[i]
        b = int4_base[i]
        s = cell_b[i]
        joined.append(
            {
                "id": i,
                "question": g["question"],
                "parametric_answer": g["parametric_answer"],
                "context_answer": g["context_answer"],
                "baseline_label": b["label"],
                "baseline_text": b["text"],
                "cell_b_label": s["label"],
                "cell_b_text": s["text"],
                "cell_b_exam_meta": exam_meta(s["text"]),
                "forward_label": forward[i]["label"],
                "forward_text": forward[i]["text"],
                "forward_exam_meta": exam_meta(forward[i]["text"]),
                "fp16_k10_label": fp16_k10[i]["label"],
                "fp16_k10_text": fp16_k10[i]["text"],
                "fp16_k10_exam_meta": exam_meta(fp16_k10[i]["text"]),
                "fp16_base_label": fp16_base[i]["label"],
                "fp16_base_text": fp16_base[i]["text"],
                "flip": f"{b['label']}→{s['label']}",
            }
        )

    ctx_items = [r for r in joined if r["cell_b_label"] == "context"]
    other_items = [r for r in joined if r["cell_b_label"] == "other"]
    par_items = [r for r in joined if r["cell_b_label"] == "parametric"]
    flips_to_ctx = [r for r in ctx_items if r["baseline_label"] != "context"]
    already_ctx = [r for r in ctx_items if r["baseline_label"] == "context"]

    def rate(rows_by_id: dict, among_label: str | None = None) -> dict:
        subset = (
            {i: r for i, r in rows_by_id.items() if r["label"] == among_label}
            if among_label is not None
            else rows_by_id
        )
        n = len(subset)
        n_em = sum(1 for r in subset.values() if exam_meta(r.get("text", "")))
        return {
            "n": n,
            "n_exam_meta": n_em,
            "rate": (n_em / n) if n else None,
        }

    exam_rates = {
        "cell_b": {
            "any_label": rate(cell_b),
            "among_context": rate(cell_b, "context"),
            "among_other": rate(cell_b, "other"),
            "among_parametric": rate(cell_b, "parametric"),
        },
        "int4_baseline": {
            "any_label": rate(int4_base),
            "among_context": rate(int4_base, "context"),
        },
        "forward_fp16_k10_on_int4": {
            "any_label": rate(forward),
            "among_context": rate(forward, "context"),
        },
        "fp16_own_k10": {
            "any_label": rate(fp16_k10),
            "among_context": rate(fp16_k10, "context"),
        },
        "fp16_baseline": {
            "any_label": rate(fp16_base),
            "among_context": rate(fp16_base, "context"),
        },
    }

    inspect_ctx = flips_to_ctx[:16]
    seen = {r["id"] for r in inspect_ctx}
    for r in ctx_items:
        if r["id"] not in seen and len(inspect_ctx) < 20:
            inspect_ctx.append(r)

    def brief(r: dict) -> dict:
        return {
            "id": r["id"],
            "question": r["question"],
            "context_gold": r["context_answer"],
            "parametric_gold": r["parametric_answer"],
            "flip": r["flip"],
            "baseline_label": r["baseline_label"],
            "baseline": r["baseline_text"],
            "cell_b_label": r["cell_b_label"],
            "cell_b": r["cell_b_text"],
            "cell_b_exam_meta": r["cell_b_exam_meta"],
            "forward_label": r["forward_label"],
            "forward": r["forward_text"],
            "forward_exam_meta": r["forward_exam_meta"],
            "fp16_own_k10_label": r["fp16_k10_label"],
            "fp16_own_k10": r["fp16_k10_text"],
            "fp16_own_k10_exam_meta": r["fp16_k10_exam_meta"],
        }

    # Side-by-side: prefer ids that are context in Cell B AND in at least one other K10 variant.
    side_ids = []
    for r in joined:
        if r["cell_b_label"] == "context" and (
            r["forward_label"] == "context" or r["fp16_k10_label"] == "context"
        ):
            side_ids.append(r["id"])
    # Fill with remaining Cell B context, then a few Cell B others, to reach ~8.
    for r in ctx_items:
        if r["id"] not in side_ids and len(side_ids) < 8:
            side_ids.append(r["id"])
    side_by_side = [brief(next(x for x in joined if x["id"] == i)) for i in side_ids[:8]]

    payload = {
        "cell": "cell_b_native_int4_k10_on_int4_ncalib128",
        "n_eval": len(joined),
        "label_counts": {
            "int4_baseline": label_counts(int4_base),
            "cell_b": label_counts(cell_b),
            "forward_fp16_k10_on_int4": label_counts(forward),
            "fp16_own_k10": label_counts(fp16_k10),
        },
        "flips": {
            "to_context": len(flips_to_ctx),
            "already_context": len(already_ctx),
            "to_other": sum(1 for r in joined if r["cell_b_label"] == "other" and r["baseline_label"] != "other"),
            "stay_other": sum(1 for r in joined if r["cell_b_label"] == "other" and r["baseline_label"] == "other"),
        },
        "exam_meta_rates": exam_rates,
        "exam_meta_heuristic": list(EXAM_META_KEYS),
        "inspection_context": [brief(r) for r in inspect_ctx],
        "inspection_other": [brief(r) for r in other_items],
        "side_by_side_three_k10": side_by_side,
        "verdict_fields": {
            "cell_b_exam_meta_among_context": exam_rates["cell_b"]["among_context"],
            "forward_exam_meta_among_context": exam_rates["forward_fp16_k10_on_int4"]["among_context"],
            "fp16_own_k10_exam_meta_among_context": exam_rates["fp16_own_k10"]["among_context"],
            "cell_a_reference_exam_meta_among_context": {
                "n": 32,
                "n_exam_meta": 13,
                "rate": 13 / 32,
                "note": "from unga-bunga-23/cell_a_diagnostics.json; α=1 Cell A",
            },
        },
    }

    (OUT_DIR / "cell_b_diagnostics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    em_ctx = exam_rates["cell_b"]["among_context"]
    em_any = exam_rates["cell_b"]["any_label"]
    em_fwd = exam_rates["forward_fp16_k10_on_int4"]["among_context"]
    em_fp16 = exam_rates["fp16_own_k10"]["among_context"]
    em_base = exam_rates["int4_baseline"]["among_context"]

    lines = [
        "# Cell B inspection (native INT4 K10 → INT4 N_CALIB=128)",
        "",
        "Same exam-meta heuristic as Cell A (`multiple-choice`, `test-taker`, `the question is`, `note:`, `this question is`, `the best answer is`, `correctness:`).",
        "",
        f"INT4 N128 baseline: P/C/O {label_counts(int4_base)['parametric']}/{label_counts(int4_base)['context']}/{label_counts(int4_base)['other']}.",
        f"Cell B: P/C/O {label_counts(cell_b)['parametric']}/{label_counts(cell_b)['context']}/{label_counts(cell_b)['other']}. Context labels: {len(ctx_items)} (flips into context: {len(flips_to_ctx)}; already context: {len(already_ctx)}).",
        f"Forward transfer (FP16 K10 → INT4): P/C/O {label_counts(forward)['parametric']}/{label_counts(forward)['context']}/{label_counts(forward)['other']}.",
        f"FP16 own K10: P/C/O {label_counts(fp16_k10)['parametric']}/{label_counts(fp16_k10)['context']}/{label_counts(fp16_k10)['other']}.",
        "",
        "## Exam-meta rates",
        "",
        f"- Cell B any label: **{em_any['n_exam_meta']}/{em_any['n']}** ({100 * em_any['rate']:.0f}%)",
        f"- Cell B among context: **{em_ctx['n_exam_meta']}/{em_ctx['n']}** ({100 * (em_ctx['rate'] or 0):.0f}%)",
        f"- INT4 baseline among context: {em_base['n_exam_meta']}/{em_base['n']} ({100 * (em_base['rate'] or 0):.0f}%)",
        f"- Forward transfer among context: **{em_fwd['n_exam_meta']}/{em_fwd['n']}** ({100 * (em_fwd['rate'] or 0):.0f}%)",
        f"- FP16 own K10 among context: **{em_fp16['n_exam_meta']}/{em_fp16['n']}** ({100 * (em_fp16['rate'] or 0):.0f}%)",
        "- Cell A (reference, α=1) among context: **13/32** (41%)",
        "",
        "## Context-labeled (flips first, then remaining to 20)",
        "",
    ]
    for r in inspect_ctx:
        lines.append(f"### {r['id']}  `{r['flip']}`  exam_meta={r['cell_b_exam_meta']}")
        lines.append(f"- Q: {r['question']}")
        lines.append(f"- context gold: {r['context_answer']}")
        lines.append(f"- parametric gold: {r['parametric_answer']}")
        lines.append(f"- INT4 baseline [{r['baseline_label']}]: {r['baseline_text']!r}")
        lines.append(f"- Cell B [{r['cell_b_label']}]: {r['cell_b_text']!r}")
        lines.append(f"- Forward FP16-K10→INT4 [{r['forward_label']}] exam_meta={r['forward_exam_meta']}: {r['forward_text']!r}")
        lines.append(f"- FP16-own K10 [{r['fp16_k10_label']}] exam_meta={r['fp16_k10_exam_meta']}: {r['fp16_k10_text']!r}")
        lines.append("")

    lines.append(f"## Other (all {len(other_items)})")
    lines.append("")
    for r in other_items:
        lines.append(f"### {r['id']}  `{r['flip']}`  exam_meta={r['cell_b_exam_meta']}")
        lines.append(f"- Q: {r['question']}")
        lines.append(f"- context gold: {r['context_answer']}")
        lines.append(f"- INT4 baseline [{r['baseline_label']}]: {r['baseline_text']!r}")
        lines.append(f"- Cell B: {r['cell_b_text']!r}")
        lines.append(f"- Forward [{r['forward_label']}]: {r['forward_text']!r}")
        lines.append("")

    lines.append("## Side-by-side: three context-K10 variants (same ids)")
    lines.append("")
    lines.append("Same handful of items that Cell B labels context, compared to FP16 own K10 and forward transfer.")
    lines.append("")
    for r in side_by_side:
        src = next(x for x in joined if x["id"] == r["id"])
        lines.append(f"### {r['id']}  Q: {src['question']}")
        lines.append(f"- (a) FP16 own K10 [{r['fp16_own_k10_label']}] exam_meta={r['fp16_own_k10_exam_meta']}: {r['fp16_own_k10']!r}")
        lines.append(f"- (b) Forward FP16-K10→INT4 [{r['forward_label']}] exam_meta={r['forward_exam_meta']}: {r['forward']!r}")
        lines.append(f"- (c) Cell B native INT4 K10 [{r['cell_b_label']}] exam_meta={r['cell_b_exam_meta']}: {r['cell_b']!r}")
        lines.append("")

    lines.append("## Qualitative notes (filled after reading texts)")
    lines.append("")
    lines.append("(Script writes the tables; verdict is written by the follow-up pass that reads this file.)")
    lines.append("")

    (OUT_DIR / "cell_b_inspection.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT_DIR / "cell_b_diagnostics.json")
    print("wrote", OUT_DIR / "cell_b_inspection.md")
    print("cell_b P/C/O", label_counts(cell_b))
    print("exam_meta any", em_any)
    print("exam_meta context", em_ctx)
    print("forward context", em_fwd)
    print("fp16 own k10 context", em_fp16)
    print("flips_to_ctx", len(flips_to_ctx), "other", len(other_items))


if __name__ == "__main__":
    main()
