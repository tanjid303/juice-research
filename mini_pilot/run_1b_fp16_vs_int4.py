#!/usr/bin/env python3
"""
Mini-pilot §5.1: Llama-3.2-1B-Instruct FP16 vs INT4 on ConflictQA.

Loads one precision at a time with explicit VRAM cleanup.
Computes MR and MR★. MR★ uses the shared double-filter set
(closed-book correct at FP16 AND at INT4) for both precisions (§5.3).

INT4 backend:
  --quant_backend gptq     (default, plan-primary; needs peft==0.10.0)
  --quant_backend bnb_nf4  (local Windows smoke only — NOT GPTQ; see NOTES.md)

Usage:
  python mini_pilot/run_1b_fp16_vs_int4.py --n_items 50 --data data/conflictqa_smoke.jsonl
  python mini_pilot/run_1b_fp16_vs_int4.py --n_items 20 --quant_backend bnb_nf4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1B mini-pilot: FP16 vs INT4 MR / MR*")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "conflictqa_smoke.jsonl")
    p.add_argument("--n_items", type=int, default=50, help="Items to evaluate (start small on 4GB)")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--quant_backend",
        choices=("gptq", "bnb_nf4"),
        default="gptq",
        help="INT4 method. gptq=plan-primary; bnb_nf4=local Windows debug only (not GPTQ).",
    )
    p.add_argument("--out_dir", type=Path, default=ROOT / "mini_pilot" / "outputs")
    p.add_argument("--n_boot", type=int, default=1000, help="Bootstrap draws for MR★ CI")
    return p.parse_args()


def run_precision(precision: str, items, max_new_tokens: int, quant_backend: str) -> dict:
    from common.inference import generate_for_item
    from common.models import load_model_and_tokenizer, unload_model, vram_mib
    from common.scoring import closed_book_correct, label_conflict_response

    backend_used = "fp16" if precision == "fp16" else quant_backend
    print(f"  Loading {precision} (backend={backend_used})...")
    model, tokenizer = load_model_and_tokenizer(
        precision,  # type: ignore[arg-type]
        quant_backend=quant_backend,  # type: ignore[arg-type]
    )
    records = []
    try:
        for i, item in enumerate(items, start=1):
            if i == 1 or i % 25 == 0 or i == len(items):
                print(f"    [{precision}] {i}/{len(items)}", flush=True)
            cb = generate_for_item(model, tokenizer, item, "closed_book", max_new_tokens=max_new_tokens)
            agrees = generate_for_item(model, tokenizer, item, "context_agrees", max_new_tokens=max_new_tokens)
            conflict = generate_for_item(
                model, tokenizer, item, "context_conflicts", max_new_tokens=max_new_tokens
            )
            label = label_conflict_response(
                conflict,
                item.parametric_answer,
                item.context_answer,
                question=item.question,
            )
            records.append(
                {
                    "id": item.id,
                    "precision": precision,
                    "quant_backend": backend_used,
                    "closed_book": cb,
                    "closed_book_correct": closed_book_correct(
                        cb, item.parametric_answer, question=item.question
                    ),
                    "context_agrees": agrees,
                    "context_conflicts": conflict,
                    "conflict_label": label,
                }
            )
    finally:
        peak = vram_mib()
        unload_model(model)

    return {"records": records, "peak_vram_mib": peak, "quant_backend": backend_used}


def summarize_mr(df: pd.DataFrame, *, quant_backend: str) -> tuple[pd.DataFrame, set[str]]:
    """
    Aggregate MR / MR★. MR★ denominator is the shared intersection:
    items with closed_book_correct at BOTH FP16 and INT4 (§5.3).
    """
    fp16_ok = set(df.loc[(df["precision"] == "fp16") & df["closed_book_correct"], "id"])
    int4_ok = set(df.loc[(df["precision"] == "int4") & df["closed_book_correct"], "id"])
    shared_ids = fp16_ok & int4_ok

    rows = []
    for precision in ("fp16", "int4"):
        all_c = df[df["precision"] == precision]
        star = all_c[all_c["id"].isin(shared_ids)]
        backend = "fp16" if precision == "fp16" else quant_backend
        cb_n = int(all_c["closed_book_correct"].sum())
        mr_num = int((all_c["conflict_label"] == "parametric").sum())
        mr_den = len(all_c)
        mrs_den = len(shared_ids)
        mrs_num = int((star["conflict_label"] == "parametric").sum()) if mrs_den else 0
        rows.append(
            {
                "precision": precision,
                "quant_backend": backend,
                "n_items": mr_den,
                "n_closed_book_correct": cb_n,
                "n_mr_star": mrs_den,
                "closed_book_acc": f"{cb_n}/{mr_den}",
                "MR": f"{mr_num}/{mr_den}",
                "MR_star": f"{mrs_num}/{mrs_den}" if mrs_den else "n/a",
                "MR_ratio": mr_num / mr_den if mr_den else float("nan"),
                "MR_star_ratio": mrs_num / mrs_den if mrs_den else float("nan"),
            }
        )
    return pd.DataFrame(rows), shared_ids


def main() -> None:
    args = parse_args()
    from common.conflictqa import load_conflict_items

    items = load_conflict_items(args.data, n_items=args.n_items)
    if not items:
        raise SystemExit(f"No items in {args.data}. See data/README.md.")

    if args.quant_backend == "bnb_nf4":
        print(
            "WARNING: --quant_backend bnb_nf4 is a LOCAL DEBUG substitute only. "
            "Do not treat results as GPTQ. Re-run with gptq on WSL/A4500 for real Fix 1 / Stage 2."
        )

    print(f"Running FP16 on {len(items)} items...")
    fp16 = run_precision("fp16", items, args.max_new_tokens, args.quant_backend)
    print(f"FP16 peak VRAM: {fp16.get('peak_vram_mib', 'n/a')} MiB")

    print(f"Running INT4 (quant_backend={args.quant_backend}) on {len(items)} items...")
    int4 = run_precision("int4", items, args.max_new_tokens, args.quant_backend)
    print(f"INT4 peak VRAM: {int4.get('peak_vram_mib', 'n/a')} MiB (backend={args.quant_backend})")

    df = pd.DataFrame(fp16["records"] + int4["records"])
    summary, shared_ids = summarize_mr(df, quant_backend=args.quant_backend)

    from common.stats_analysis import analyze_behavioral

    stats = analyze_behavioral(df, n_boot=args.n_boot, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = args.out_dir / f"summary_{stamp}.csv"
    detail_path = args.out_dir / f"detail_{stamp}.jsonl"
    meta_path = args.out_dir / f"run_meta_{stamp}.json"
    stats_path = args.out_dir / f"stats_{stamp}.json"
    summary.to_csv(summary_path, index=False)
    with detail_path.open("w", encoding="utf-8") as f:
        for r in fp16["records"] + int4["records"]:
            f.write(json.dumps(r) + "\n")
    meta = {
        "n_items": len(items),
        "data": str(args.data),
        "quant_backend_int4": args.quant_backend,
        "fp16_peak_vram_mib": fp16.get("peak_vram_mib"),
        "int4_peak_vram_mib": int4.get("peak_vram_mib"),
        "mr_star_shared_n": len(shared_ids),
        "mr_star_filter": "FP16_CB_correct AND INT4_CB_correct (shared intersection)",
        "n_boot": args.n_boot,
        "note": (
            "bnb_nf4 is local-debug-only; GPTQ is plan-primary for A4500/Fix1/Stage2"
            if args.quant_backend == "bnb_nf4"
            else "GPTQ-INT4 (plan-primary)"
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(summary))
    width = 0.35
    ax.bar([i - width / 2 for i in x], summary["MR_ratio"], width, label="MR")
    ax.bar([i + width / 2 for i in x], summary["MR_star_ratio"], width, label="MR★")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{p}\n({b})" for p, b in zip(summary["precision"], summary["quant_backend"])])
    ax.set_ylabel("Memorization ratio")
    ax.set_title(f"Mini-pilot n={len(items)} INT4={args.quant_backend}")
    ax.legend()
    ax.set_ylim(0, 1)
    plot_path = args.out_dir / f"mr_vs_mrstar_{stamp}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    boot = stats["bootstrap_mr_star"]
    ame = stats["logistic_mixed_ame"]
    print("\n=== Bootstrap MR★ (shared double-filter, 95% percentile CI) ===")
    if boot.get("n_shared", 0):
        d = boot["delta_int4_minus_fp16"]
        print(f"  n_shared={boot['n_shared']}  n_boot={boot['n_boot']}")
        print(
            f"  FP16 MR★={boot['fp16_mr_star']['point']:.4f} "
            f"CI95=[{boot['fp16_mr_star']['ci95'][0]:.4f}, {boot['fp16_mr_star']['ci95'][1]:.4f}]"
        )
        print(
            f"  INT4 MR★={boot['int4_mr_star']['point']:.4f} "
            f"CI95=[{boot['int4_mr_star']['ci95'][0]:.4f}, {boot['int4_mr_star']['ci95'][1]:.4f}]"
        )
        print(
            f"  Δ (INT4-FP16)={d['point_pp']:+.2f} pp "
            f"CI95=[{d['ci95_pp'][0]:+.2f}, {d['ci95_pp'][1]:+.2f}] pp "
            f"excludes_0={d['excludes_zero']}"
        )
    print("\n=== Logistic mixed AME on follows_context (INT4 vs FP16) ===")
    if "ame_pp" in ame:
        print(f"  method={ame.get('method')}")
        print(
            f"  AME={ame['ame_pp']:+.2f} pp "
            f"CI95=[{ame['ame_ci95_pp'][0]:+.2f}, {ame['ame_ci95_pp'][1]:+.2f}] pp "
            f"excludes_0={ame.get('ci_excludes_zero')}"
        )
        if ame.get("note"):
            print(f"  note: {ame['note']}")
    else:
        print(f"  failed: {ame}")
    print(f"\nINT4 quant_backend used: {args.quant_backend}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
