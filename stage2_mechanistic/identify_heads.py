#!/usr/bin/env python3
"""Identify reliable JuICE heads (faithful dual-run; K=10 for 7–8B)."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JuICE head identification (real dual-run)")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "conflictqa_smoke.jsonl")
    p.add_argument("--n_items", type=int, default=50)
    p.add_argument("--n_calib", type=int, default=None, help="Calibration subset (default: min(32, n_items))")
    p.add_argument("--precision", choices=("fp16", "int4"), default="fp16")
    p.add_argument("--K", type=int, default=10, help="K=10 for Llama-3 class; K=5 for smaller")
    p.add_argument("--steer_target", choices=("context", "parametric", "both"), default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ROOT / "stage2_mechanistic" / "outputs" / "heads_fp16.json")
    p.add_argument("--out_parametric", type=Path, default=None)
    p.add_argument("--device_map", type=str, default=None, help='e.g. "auto" for 8B FP16 on T4x2')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from common.conflictqa import load_conflict_items
    from common.juice import (
        identify_both_directions,
        identify_reliable_heads,
        print_identified,
        save_identified_heads,
    )
    from common.models import load_model_and_tokenizer, unload_model

    items = load_conflict_items(args.data, n_items=args.n_items)
    n_calib = args.n_calib or min(32, len(items))
    calib = items[:n_calib]

    load_kw = {}
    if args.device_map:
        load_kw["device_map"] = args.device_map
        load_kw["forbid_cpu_offload"] = True
    model, tokenizer = load_model_and_tokenizer(args.precision, **load_kw)  # type: ignore[arg-type]
    try:
        if args.steer_target == "both":
            ctx, par = identify_both_directions(
                model,
                tokenizer,
                calib,
                precision=args.precision,
                K=args.K,
                seed=args.seed,
                checkpoint_dir=args.out.parent / "id_checkpoints",
            )
            save_identified_heads(args.out, ctx)
            par_path = args.out_parametric or args.out.with_name(
                args.out.stem.replace("context", "parametric") + "_parametric.json"
                if "context" in args.out.stem
                else args.out.stem + "_parametric.json"
            )
            if par_path == args.out:
                par_path = args.out.with_name(args.out.stem + "_parametric.json")
            save_identified_heads(par_path, par)
            print_identified(ctx)
            print_identified(par)
            print(f"Saved context heads -> {args.out}")
            print(f"Saved parametric heads -> {par_path}")
        else:
            ident = identify_reliable_heads(
                model,
                tokenizer,
                calib,
                precision=args.precision,
                K=args.K,
                steer_target=args.steer_target,
                seed=args.seed,
            )
            save_identified_heads(args.out, ident)
            print_identified(ident)
            print(f"Saved {len(ident.enlarge)}/+ {len(ident.suppress)}/− heads to {args.out}")
    finally:
        unload_model(model)


if __name__ == "__main__":
    main()
