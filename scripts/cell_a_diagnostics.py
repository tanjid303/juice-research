"""Local diagnostics for Cell A (INT4 K10 → FP16): text inspection + par_pos/ctx_pos.

No GPU. Writes:
  unga-bunga-23/cell_a_diagnostics.json
  unga-bunga-23/cell_a_inspection.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"d:\tt")
STEERED = ROOT / "unga-bunga-23" / "juice_stage2_int4_ncalib128" / "steer_reverse_int4_context_k10_on_fp16.json"
BASE = ROOT / "juice_stage2_base_directional_cell3" / "steer_baseline.json"
FP16_K10_STEER = ROOT / "juice_stage2_base_directional_cell3" / "steer_context_k10.json"
DATA = ROOT / "data" / "conflictqa_pilot250_popular.jsonl"
INT4_DIAG = ROOT / "juice_stage2_int4_native_id" / "id_diagnostics_raw.json"
FP16_DIAG = ROOT / "juice_stage2_base_id" / "id_diagnostics_raw.json"
OUT_DIR = ROOT / "unga-bunga-23"

INT4_K10 = [
    "L7_H26", "L14_H10", "L7_H21", "L4_H5", "L3_H3",
    "L31_H20", "L4_H11", "L14_H23", "L8_H29", "L9_H0",
]
FP16_K10 = [
    "L4_H9", "L9_H0", "L1_H11", "L4_H5", "L19_H27",
    "L17_H12", "L7_H26", "L14_H23", "L3_H22", "L31_H20",
]


def load_jsonl(path: Path) -> dict:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[obj["id"]] = obj
    return out


def scores_for(diag: dict, heads: list[str]) -> list[dict]:
    ctx = diag["scores_pos_context"]
    par = diag["scores_pos_parametric"]
    rows = []
    for h in heads:
        c = float(ctx.get(h, 0.0))
        p = float(par.get(h, 0.0))
        rows.append(
            {
                "head": h,
                "ctx_pos": c,
                "par_pos": p,
                "directional_ctx": c - p,
                "both_positive": c > 0 and p > 0,
                "universal_booster_flag": c > 0 and p > 0,
                "context_specific": c > 0 and p <= 0,
            }
        )
    return rows


def main() -> None:
    gold = load_jsonl(DATA)
    steered = {r["id"]: r for r in json.loads(STEERED.read_text(encoding="utf-8"))}
    base = {r["id"]: r for r in json.loads(BASE.read_text(encoding="utf-8"))}
    fp16_k10 = {r["id"]: r for r in json.loads(FP16_K10_STEER.read_text(encoding="utf-8"))}

    int4_diag = json.loads(INT4_DIAG.read_text(encoding="utf-8"))
    fp16_diag = json.loads(FP16_DIAG.read_text(encoding="utf-8"))

    joined = []
    for i in steered:
        g = gold[i]
        b = base[i]
        s = steered[i]
        joined.append(
            {
                "id": i,
                "question": g["question"],
                "parametric_answer": g["parametric_answer"],
                "context_answer": g["context_answer"],
                "baseline_label": b["label"],
                "baseline_text": b["text"],
                "cell_a_label": s["label"],
                "cell_a_text": s["text"],
                "fp16_k10_label": fp16_k10[i]["label"],
                "fp16_k10_text": fp16_k10[i]["text"],
                "flip": f"{b['label']}→{s['label']}",
            }
        )

    ctx_items = [r for r in joined if r["cell_a_label"] == "context"]
    other_items = [r for r in joined if r["cell_a_label"] == "other"]
    par_items = [r for r in joined if r["cell_a_label"] == "parametric"]
    flips_to_ctx = [r for r in ctx_items if r["baseline_label"] != "context"]

    exam_meta = []
    for r in joined:
        t = r["cell_a_text"].lower()
        if any(
            k in t
            for k in (
                "multiple-choice",
                "test-taker",
                "the question is",
                "note:",
                "this question is",
                "the best answer is",
                "correctness:",
            )
        ):
            exam_meta.append(r["id"])

    # Prefer flips for inspection, then remaining context.
    inspect_ctx = flips_to_ctx[:16]
    seen = {r["id"] for r in inspect_ctx}
    for r in ctx_items:
        if r["id"] not in seen and len(inspect_ctx) < 20:
            inspect_ctx.append(r)

    def brief(r, n=18):
        return {
            "id": r["id"],
            "question": r["question"],
            "context_gold": r["context_answer"],
            "parametric_gold": r["parametric_answer"],
            "flip": r["flip"],
            "baseline": r["baseline_text"],
            "cell_a": r["cell_a_text"],
            "fp16_own_k10": r["fp16_k10_text"],
            "fp16_own_k10_label": r["fp16_k10_label"],
            "exam_meta": r["id"] in exam_meta,
        }

    payload = {
        "counts": {
            "cell_a": {
                "context": len(ctx_items),
                "parametric": len(par_items),
                "other": len(other_items),
                "rejects_context": sum(1 for r in joined if r["cell_a_label"] == "rejects_context"),
            },
            "flips_to_context": len(flips_to_ctx),
            "already_context": sum(1 for r in ctx_items if r["baseline_label"] == "context"),
            "exam_meta_any_label": len(exam_meta),
            "exam_meta_among_context": sum(1 for r in ctx_items if r["id"] in exam_meta),
        },
        "inspection_context": [brief(r) for r in inspect_ctx],
        "inspection_other": [brief(r) for r in other_items],
        "inspection_parametric": [brief(r) for r in par_items],
        "int4_k10_on_int4_id": scores_for(int4_diag, INT4_K10),
        "int4_k10_on_fp16_id": scores_for(fp16_diag, INT4_K10),
        "fp16_k10_on_fp16_id": scores_for(fp16_diag, FP16_K10),
        "fp16_k10_on_int4_id": scores_for(int4_diag, FP16_K10),
    }

    n_univ_int4 = sum(1 for r in payload["int4_k10_on_int4_id"] if r["both_positive"])
    n_univ_fp16ref = sum(1 for r in payload["fp16_k10_on_fp16_id"] if r["both_positive"])
    payload["universal_booster_verdict"] = {
        "int4_k10_heads_both_pos_on_int4_id": n_univ_int4,
        "fp16_k10_heads_both_pos_on_fp16_id": n_univ_fp16ref,
        "note": (
            "Universal-booster ID signature = ctx_pos>0 AND par_pos>0. "
            f"INT4 K10: {n_univ_int4}/10. FP16 K10 reference: {n_univ_fp16ref}/10."
        ),
    }

    (OUT_DIR / "cell_a_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Cell A inspection (INT4 K10 → FP16)", ""]
    lines.append(f"Context labels: {len(ctx_items)} (flips into context: {len(flips_to_ctx)})")
    lines.append(f"Exam-meta phrasing in Cell A text: {len(exam_meta)}/50 items, {sum(1 for r in ctx_items if r['id'] in exam_meta)}/{len(ctx_items)} of context labels.")
    lines.append("")
    lines.append("## Context-labeled (mostly parametric→context flips)")
    for r in inspect_ctx:
        lines.append(f"### {r['id']}  `{r['flip']}`  exam_meta={r['id'] in exam_meta}")
        lines.append(f"- Q: {r['question']}")
        lines.append(f"- context gold: {r['context_answer']}")
        lines.append(f"- parametric gold: {r['parametric_answer']}")
        lines.append(f"- baseline [{r['baseline_label']}]: {r['baseline_text']!r}")
        lines.append(f"- Cell A [{r['cell_a_label']}]: {r['cell_a_text']!r}")
        lines.append(f"- FP16-own K10 [{r['fp16_k10_label']}]: {r['fp16_k10_text']!r}")
        lines.append("")
    lines.append("## Other (all 6)")
    for r in other_items:
        lines.append(f"### {r['id']}  `{r['flip']}`")
        lines.append(f"- Q: {r['question']}")
        lines.append(f"- context gold: {r['context_answer']}")
        lines.append(f"- baseline [{r['baseline_label']}]: {r['baseline_text']!r}")
        lines.append(f"- Cell A: {r['cell_a_text']!r}")
        lines.append("")
    lines.append("## Remaining parametric")
    for r in par_items:
        lines.append(f"### {r['id']}  `{r['flip']}`")
        lines.append(f"- Q: {r['question']}")
        lines.append(f"- parametric gold: {r['parametric_answer']} | context gold: {r['context_answer']}")
        lines.append(f"- baseline [{r['baseline_label']}]: {r['baseline_text']!r}")
        lines.append(f"- Cell A: {r['cell_a_text']!r}")
        lines.append("")

    (OUT_DIR / "cell_a_inspection.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT_DIR / "cell_a_diagnostics.json")
    print("wrote", OUT_DIR / "cell_a_inspection.md")
    print("exam_meta", len(exam_meta), "context", len(ctx_items), "flips", len(flips_to_ctx))
    print("INT4 K10 both_pos", n_univ_int4, "FP16 K10 both_pos", n_univ_fp16ref)
    print("INT4 K10 on INT4 ID:")
    for r in payload["int4_k10_on_int4_id"]:
        print(f"  {r['head']:8s} ctx={r['ctx_pos']:+.4f} par={r['par_pos']:+.4f} dir={r['directional_ctx']:+.4f} spec={r['context_specific']}")


if __name__ == "__main__":
    main()
