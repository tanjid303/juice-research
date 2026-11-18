#!/usr/bin/env python3
"""
Bootstrap FP16 head identification; compute Jaccard overlap vs noise floor.

Resamples calibration items with replacement for each draw.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap Jaccard head drift (1B)")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "conflictqa_smoke.jsonl")
    p.add_argument("--n_items", type=int, default=50)
    p.add_argument("--n_calib", type=int, default=32)
    p.add_argument("--bootstrap_draws", type=int, default=5, help="Start low on 4GB; scale to 20+ on A4500")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare_precision", choices=("fp16", "int4"), default="int4")
    p.add_argument("--out_dir", type=Path, default=ROOT / "stage2_mechanistic" / "outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from common.conflictqa import load_conflict_items
    from common.juice import identify_reliable_heads, jaccard, save_identified_heads
    from common.models import load_model_and_tokenizer, unload_model

    items = load_conflict_items(args.data, n_items=args.n_items)
    pool = items[: min(args.n_calib, len(items))]
    rng = random.Random(args.seed)

    # Reference FP16 identification on full calib pool
    model_fp16, tok = load_model_and_tokenizer("fp16")
    try:
        ref = identify_reliable_heads(model_fp16, tok, pool, precision="fp16", K=args.K, seed=args.seed)
    finally:
        unload_model(model_fp16)

    bootstrap_sets = []
    for b in range(args.bootstrap_draws):
        sample = [pool[rng.randrange(len(pool))] for _ in range(len(pool))]
        model, tokenizer = load_model_and_tokenizer("fp16")
        try:
            ident = identify_reliable_heads(
                model, tokenizer, sample, precision="fp16", K=args.K, seed=args.seed + b + 1
            )
        finally:
            unload_model(model)
        bootstrap_sets.append(ident.heads)

    boot_jaccards = [jaccard(ref.heads, bs) for bs in bootstrap_sets]
    noise_floor = float(np.percentile(boot_jaccards, 5)) if boot_jaccards else 0.0

    # Cross-precision: FP16 ref vs INT4 native ID
    model_int4, tok4 = load_model_and_tokenizer("int4")
    try:
        native_int4 = identify_reliable_heads(
            model_int4, tok4, pool, precision="int4", K=args.K, seed=args.seed
        )
    finally:
        unload_model(model_int4)

    cross_j = jaccard(ref.heads, native_int4.heads)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_identified_heads(args.out_dir / f"heads_fp16_ref_{stamp}.json", ref)
    save_identified_heads(args.out_dir / f"heads_int4_native_{stamp}.json", native_int4)

    report = {
        "ref_heads": [h.key() for h in ref.heads],
        "native_int4_heads": [h.key() for h in native_int4.heads],
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_jaccards": boot_jaccards,
        "bootstrap_jaccard_mean": float(np.mean(boot_jaccards)) if boot_jaccards else None,
        "noise_floor_p5": noise_floor,
        "cross_precision_jaccard_fp16_vs_int4": cross_j,
        "below_noise_floor": cross_j < noise_floor,
    }
    out_path = args.out_dir / f"jaccard_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Bootstrap Jaccards: {boot_jaccards}")
    print(f"Noise floor (5th pct): {noise_floor:.3f}")
    print(f"FP16 vs INT4 Jaccard: {cross_j:.3f} (below floor={report['below_noise_floor']})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
