#!/usr/bin/env python3
"""
Mini-pilot §5.1 on Llama-3.1-8B-Instruct: FP16 vs GPTQ-INT4.

Same analysis as mini_pilot/run_1b_fp16_vs_int4.py (shared double-filter MR★,
bootstrap CI, mixed-logit AME). Adds:

  - device_map="auto" for FP16 (required on Kaggle T4 x2; ~16.1 GiB weights)
  - GPTQ pinned to GPU 0 (fits a single 16GB T4; kernels happier that way)
  - JSONL checkpoint every --checkpoint_every items, resumable after a kill

Usage (Kaggle / 2x T4):
  python mini_pilot/run_8b_fp16_vs_int4.py --n_items 250 --data data/conflictqa_pilot250_popular.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.models import (
    FP16_8B_MODEL_ID,
    GPTQ_8B_MODEL_ID,
    load_model_and_tokenizer,
    unload_model,
    vram_mib,
    vram_per_device_mib,
)
from mini_pilot.checkpoint import (
    atomic_write_jsonl,
    done_ids,
    find_resume_checkpoint,
    load_jsonl,
    write_meta,
)


def _load_summarize_mr():
    """Same summarize_mr as the 1B runner (import by path; no package install needed)."""
    path = ROOT / "mini_pilot" / "run_1b_fp16_vs_int4.py"
    spec = importlib.util.spec_from_file_location("run_1b_fp16_vs_int4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.summarize_mr


summarize_mr = _load_summarize_mr()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="8B mini-pilot: FP16 vs GPTQ-INT4 MR / MR*")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "conflictqa_pilot250_popular.jsonl")
    p.add_argument("--n_items", type=int, default=50, help="Items to evaluate (same flag as the 1B runner)")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--quant_backend",
        choices=("gptq", "bnb_nf4"),
        default="gptq",
        help="INT4 method. gptq=plan-primary; bnb_nf4=local debug only (not GPTQ).",
    )
    p.add_argument("--out_dir", type=Path, default=ROOT / "mini_pilot" / "outputs")
    p.add_argument("--n_boot", type=int, default=1000, help="Bootstrap draws for MR★ CI")
    p.add_argument("--model_id_fp16", default=FP16_8B_MODEL_ID)
    p.add_argument("--model_id_int4", default=GPTQ_8B_MODEL_ID)
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=20,
        help="Flush generations to JSONL every N completed items of the current precision",
    )
    p.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help="JSONL checkpoint (default: <out_dir>/checkpoint_8b.jsonl)",
    )
    p.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Optional extra JSONL (e.g. /kaggle/input/.../checkpoint_8b.jsonl)",
    )
    p.add_argument(
        "--fp16_device_map",
        default="auto",
        help='FP16 placement. "auto" shards across T4 x2.',
    )
    p.add_argument(
        "--int4_device_map",
        default="gpu0",
        choices=("gpu0", "auto"),
        help='GPTQ placement. gpu0 pins {"": 0} (recommended). auto is supported but unnecessary.',
    )
    return p.parse_args()


def default_args(**overrides) -> argparse.Namespace:
    """Build the same Namespace as the CLI, for Kaggle notebook cells."""
    ns = argparse.Namespace(
        data=ROOT / "data" / "conflictqa_pilot250_popular.jsonl",
        n_items=50,
        max_new_tokens=32,
        seed=42,
        quant_backend="gptq",
        out_dir=ROOT / "mini_pilot" / "outputs",
        n_boot=1000,
        model_id_fp16=FP16_8B_MODEL_ID,
        model_id_int4=GPTQ_8B_MODEL_ID,
        checkpoint_every=20,
        checkpoint_path=None,
        resume_from=None,
        fp16_device_map="auto",
        int4_device_map="gpu0",
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _device_map_arg(spec: str):
    if spec in ("auto", "cpu"):
        return spec
    if spec == "gpu0":
        return {"": 0}
    return spec


def _save_ckpt(path: Path, meta_path: Path, records: list[dict], extra: dict) -> None:
    atomic_write_jsonl(path, records)
    n_fp = sum(1 for r in records if r.get("precision") == "fp16")
    n_i4 = sum(1 for r in records if r.get("precision") == "int4")
    write_meta(
        meta_path,
        {"n_records": len(records), "n_fp16": n_fp, "n_int4": n_i4, **extra},
    )
    print(f"    checkpoint {path.name}: fp16={n_fp} int4={n_i4}", flush=True)


def run_precision(
    precision: str,
    items,
    max_new_tokens: int,
    quant_backend: str,
    *,
    model_id_fp16: str,
    model_id_int4: str,
    device_map,
    all_records: list[dict],
    checkpoint_path: Path,
    meta_path: Path,
    checkpoint_every: int,
    forbid_cpu_offload: bool,
) -> dict:
    from common.inference import generate_for_item
    from common.scoring import closed_book_correct, label_conflict_response

    already = done_ids(all_records, precision)
    pending = [it for it in items if it.id not in already]
    backend_used = "fp16" if precision == "fp16" else quant_backend
    print(
        f"  {precision}: {len(already)} already in checkpoint, {len(pending)} remaining "
        f"(backend={backend_used}, device_map={device_map!r})"
    )
    if not pending:
        return {
            "records": [r for r in all_records if r.get("precision") == precision],
            "peak_vram_mib": None,
            "vram_per_device_mib": {},
            "quant_backend": backend_used,
            "skipped_load": True,
        }

    print(f"  Loading {precision} ({model_id_fp16 if precision == 'fp16' else model_id_int4})...")
    model, tokenizer = load_model_and_tokenizer(
        precision,  # type: ignore[arg-type]
        model_id_fp16=model_id_fp16,
        model_id_int4=model_id_int4,
        quant_backend=quant_backend,  # type: ignore[arg-type]
        device_map=device_map,
        forbid_cpu_offload=forbid_cpu_offload,
    )
    print(f"  hf_device_map: {getattr(model, 'hf_device_map', None)}", flush=True)

    new_n = 0
    try:
        try:
            for i, item in enumerate(pending, start=1):
                if i == 1 or i % 25 == 0 or i == len(pending):
                    print(f"    [{precision}] {i}/{len(pending)} (id={item.id})", flush=True)
                cb = generate_for_item(
                    model, tokenizer, item, "closed_book", max_new_tokens=max_new_tokens
                )
                agrees = generate_for_item(
                    model, tokenizer, item, "context_agrees", max_new_tokens=max_new_tokens
                )
                conflict = generate_for_item(
                    model, tokenizer, item, "context_conflicts", max_new_tokens=max_new_tokens
                )
                label = label_conflict_response(
                    conflict,
                    item.parametric_answer,
                    item.context_answer,
                    question=item.question,
                )
                all_records.append(
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
                new_n += 1
                if checkpoint_every > 0 and new_n % checkpoint_every == 0:
                    _save_ckpt(
                        checkpoint_path,
                        meta_path,
                        all_records,
                        {"precision_in_progress": precision, "last_id": item.id},
                    )
        except (KeyboardInterrupt, Exception):
            _save_ckpt(
                checkpoint_path,
                meta_path,
                all_records,
                {"precision_in_progress": precision, "interrupted": True},
            )
            raise
        _save_ckpt(
            checkpoint_path,
            meta_path,
            all_records,
            {"precision_in_progress": None, "just_finished": precision},
        )
    finally:
        peak = vram_mib()
        per = vram_per_device_mib()
        print(f"  {precision} peak VRAM: {peak} MiB per-device={per}")
        unload_model(model)

    return {
        "records": [r for r in all_records if r.get("precision") == precision],
        "peak_vram_mib": peak,
        "vram_per_device_mib": per,
        "quant_backend": backend_used,
        "skipped_load": False,
    }


def run_pilot(args: argparse.Namespace) -> None:
    import torch

    from common.conflictqa import load_conflict_items
    from common.stats_analysis import analyze_behavioral

    items = load_conflict_items(args.data, n_items=args.n_items)
    if not items:
        raise SystemExit(f"No items in {args.data}. See data/README.md.")

    if args.quant_backend == "bnb_nf4":
        print(
            "WARNING: --quant_backend bnb_nf4 is a LOCAL DEBUG substitute only. "
            "Do not treat results as GPTQ. Re-run with gptq for real Fix 1 / Stage 2."
        )

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"CUDA devices: {n_gpu}")
    for i in range(n_gpu):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name} {props.total_memory / 1024**3:.1f} GiB")
    if args.fp16_device_map == "auto" and n_gpu < 2:
        print(
            "WARNING: Llama-3.1-8B FP16 weights are ~16.1 GiB and will not fit on a "
            "single 16GB T4. device_map='auto' on one GPU will OOM (CPU offload is forbidden). "
            "Use a T4 x2 (or larger) session."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.checkpoint_path or (args.out_dir / "checkpoint_8b.jsonl")
    meta_path = ckpt_path.with_name(ckpt_path.stem + "_meta.json")

    extra_roots: list[Path] = []
    if args.resume_from is not None:
        extra_roots.append(args.resume_from)
    extra_roots.append(Path("/kaggle/input"))
    found = find_resume_checkpoint(ckpt_path, extra_roots)
    if found is not None and found.resolve() != ckpt_path.resolve():
        print(f"Seeding checkpoint from {found}")
        atomic_write_jsonl(ckpt_path, load_jsonl(found))
    all_records = load_jsonl(ckpt_path)
    print(f"Checkpoint {ckpt_path}: {len(all_records)} existing rows")

    fp16_map = _device_map_arg(args.fp16_device_map)
    int4_map = _device_map_arg(args.int4_device_map)

    print(f"Running FP16 on {len(items)} items...")
    fp16 = run_precision(
        "fp16",
        items,
        args.max_new_tokens,
        args.quant_backend,
        model_id_fp16=args.model_id_fp16,
        model_id_int4=args.model_id_int4,
        device_map=fp16_map,
        all_records=all_records,
        checkpoint_path=ckpt_path,
        meta_path=meta_path,
        checkpoint_every=args.checkpoint_every,
        forbid_cpu_offload=True,
    )
    print(f"FP16 peak VRAM: {fp16.get('peak_vram_mib', 'n/a')} MiB")

    print(f"Running INT4 (quant_backend={args.quant_backend}) on {len(items)} items...")
    int4 = run_precision(
        "int4",
        items,
        args.max_new_tokens,
        args.quant_backend,
        model_id_fp16=args.model_id_fp16,
        model_id_int4=args.model_id_int4,
        device_map=int4_map,
        all_records=all_records,
        checkpoint_path=ckpt_path,
        meta_path=meta_path,
        checkpoint_every=args.checkpoint_every,
        forbid_cpu_offload=False,
    )
    print(f"INT4 peak VRAM: {int4.get('peak_vram_mib', 'n/a')} MiB (backend={args.quant_backend})")

    df = pd.DataFrame(fp16["records"] + int4["records"])
    if df.empty or set(df["precision"]) != {"fp16", "int4"}:
        have = sorted(df["precision"].unique()) if not df.empty else []
        raise SystemExit(
            "Both precisions must be complete before analysis. "
            f"Have precisions={have}. Re-run this script to resume from the checkpoint."
        )
    n_fp = int((df.precision == "fp16").sum())
    n_i4 = int((df.precision == "int4").sum())
    if n_fp != n_i4 or n_fp != len(items):
        raise SystemExit(
            f"Incomplete: fp16={n_fp} int4={n_i4} expected={len(items)}. Re-run to resume."
        )

    summary, shared_ids = summarize_mr(df, quant_backend=args.quant_backend)
    stats = analyze_behavioral(df, n_boot=args.n_boot, seed=args.seed)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = args.out_dir / f"summary_{stamp}.csv"
    detail_path = args.out_dir / f"detail_{stamp}.jsonl"
    meta_out = args.out_dir / f"run_meta_{stamp}.json"
    stats_path = args.out_dir / f"stats_{stamp}.json"
    summary.to_csv(summary_path, index=False)
    atomic_write_jsonl(detail_path, fp16["records"] + int4["records"])
    meta = {
        "n_items": len(items),
        "data": str(args.data),
        "model_id_fp16": args.model_id_fp16,
        "model_id_int4": args.model_id_int4,
        "fp16_device_map": str(fp16_map),
        "int4_device_map": str(int4_map),
        "quant_backend_int4": args.quant_backend,
        "fp16_peak_vram_mib": fp16.get("peak_vram_mib"),
        "int4_peak_vram_mib": int4.get("peak_vram_mib"),
        "fp16_vram_per_device_mib": fp16.get("vram_per_device_mib"),
        "int4_vram_per_device_mib": int4.get("vram_per_device_mib"),
        "mr_star_shared_n": len(shared_ids),
        "mr_star_filter": "FP16_CB_correct AND INT4_CB_correct (shared intersection)",
        "n_boot": args.n_boot,
        "note": (
            "bnb_nf4 is local-debug-only; GPTQ is plan-primary for A4500/Fix1/Stage2"
            if args.quant_backend == "bnb_nf4"
            else "GPTQ-INT4 (plan-primary) on Llama-3.1-8B-Instruct"
        ),
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(summary))
    width = 0.35
    ax.bar([i - width / 2 for i in x], summary["MR_ratio"], width, label="MR")
    ax.bar([i + width / 2 for i in x], summary["MR_star_ratio"], width, label="MR★")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{p}\n({b})" for p, b in zip(summary["precision"], summary["quant_backend"])]
    )
    ax.set_ylabel("Memorization ratio")
    ax.set_title(f"8B mini-pilot n={len(items)} INT4={args.quant_backend}")
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
    print(f"Wrote {meta_out}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {plot_path}")
    print(f"Checkpoint remains at {ckpt_path} (download this if the session is about to die)")


def main() -> None:
    run_pilot(parse_args())


if __name__ == "__main__":
    main()
