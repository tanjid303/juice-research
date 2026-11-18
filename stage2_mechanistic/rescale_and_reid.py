#!/usr/bin/env python3
"""
Fix 1 repair experiment on 1B: rescale-only grid vs re-ID-32 vs transferred baseline.

Includes:
- Bootstrap CI on repair deltas (re-ID-32 − transferred, rescale-only − transferred)
- Negative control: re-ID on 32 random non-conflict trivia items

Usage:
  python stage2_mechanistic/rescale_and_reid.py --n_items 50 --bootstrap_draws 5
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

# Task-irrelevant calibration pool for negative control (no knowledge conflict).
NEGATIVE_CALIB = [
    {"question": "What is 2+2?", "parametric_answer": "4", "context_answer": "5",
     "passage_agrees": "Basic arithmetic: 2+2 equals 4.", "passage_conflicts": "Basic arithmetic: 2+2 equals 5."},
    {"question": "What color is the sky on a clear day?", "parametric_answer": "blue", "context_answer": "green",
     "passage_agrees": "The sky appears blue due to Rayleigh scattering.", "passage_conflicts": "The sky appears green on a clear day."},
    {"question": "How many days are in a week?", "parametric_answer": "seven", "context_answer": "eight",
     "passage_agrees": "A week has seven days.", "passage_conflicts": "A week has eight days."},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix 1 repair pilot (1B)")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "conflictqa_smoke.jsonl")
    p.add_argument("--n_items", type=int, default=50)
    p.add_argument("--repair_n", type=int, default=32, help="Items for rescale / re-ID grid")
    p.add_argument("--bootstrap_draws", type=int, default=5)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--alphas", type=str, default="0.25,0.5,1.0,1.5,2.0")
    p.add_argument("--apply_precision", choices=("int4",), default="int4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=Path, default=ROOT / "stage2_mechanistic" / "outputs")
    return p.parse_args()


def _items_from_neg(n: int, seed: int):
    from common.prompts import ConflictItem

    rng = random.Random(seed)
    out = []
    for i in range(n):
        row = rng.choice(NEGATIVE_CALIB)
        out.append(
            ConflictItem(
                id=f"neg_{i}",
                question=row["question"],
                parametric_answer=row["parametric_answer"],
                context_answer=row["context_answer"],
                passage_agrees=row["passage_agrees"],
                passage_conflicts=row["passage_conflicts"],
                source="negative_control",
            )
        )
    return out


def efficacy_on(model, tokenizer, items, heads, alpha, steer_target, max_new_tokens=16):
    from common.juice import IdentifiedHeads, apply_juice_steering, steering_efficacy

    ident = IdentifiedHeads(
        heads=heads,
        scores={},
        precision="int4",
        K=len(heads),
        steer_target=steer_target,
        n_calibration=0,
    )
    res = apply_juice_steering(model, tokenizer, items, ident, alpha=alpha, max_new_tokens=max_new_tokens)
    return steering_efficacy(res, steer_target)


def bootstrap_ci(deltas: list[float]) -> tuple[float, float, float]:
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(deltas)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    args = parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    from common.conflictqa import load_conflict_items
    from common.juice import identify_reliable_heads, load_identified_heads, save_identified_heads
    from common.models import load_model_and_tokenizer, unload_model

    eval_items = load_conflict_items(args.data, n_items=args.n_items)
    repair_pool = eval_items[: min(args.repair_n, len(eval_items))]

    # Transferred: FP16 heads @ INT4
    model_fp16, tok = load_model_and_tokenizer("fp16")
    try:
        fp16_heads = identify_reliable_heads(
            model_fp16, tok, repair_pool, precision="fp16", K=args.K, seed=args.seed
        )
    finally:
        unload_model(model_fp16)

    model, tokenizer = load_model_and_tokenizer("int4")
    try:
        transferred = efficacy_on(
            model, tokenizer, eval_items, fp16_heads.heads, alpha=1.0, steer_target=fp16_heads.steer_target
        )

        # Rescale-only grid (FP16 heads, sweep alpha)
        best_alpha, best_rescale = 1.0, transferred
        for a in alphas:
            e = efficacy_on(model, tokenizer, eval_items, fp16_heads.heads, alpha=a, steer_target=fp16_heads.steer_target)
            if e > best_rescale:
                best_rescale, best_alpha = e, a

        # Re-ID-32 on INT4
        reid = identify_reliable_heads(
            model, tokenizer, repair_pool, precision="int4", K=args.K, seed=args.seed
        )
        reid_eff = efficacy_on(
            model, tokenizer, eval_items, reid.heads, alpha=1.0, steer_target=reid.steer_target
        )

        # Negative control: re-ID on irrelevant items
        neg_pool = _items_from_neg(args.repair_n, args.seed + 99)
        neg_reid = identify_reliable_heads(
            model, tokenizer, neg_pool, precision="int4", K=args.K, seed=args.seed + 100
        )
        neg_eff = efficacy_on(
            model, tokenizer, eval_items, neg_reid.heads, alpha=1.0, steer_target=neg_reid.steer_target
        )

        # Bootstrap CIs on repair deltas
        rng = random.Random(args.seed)
        reid_deltas, rescale_deltas = [], []
        for b in range(args.bootstrap_draws):
            sample = [repair_pool[rng.randrange(len(repair_pool))] for _ in range(len(repair_pool))]
            reid_b = identify_reliable_heads(
                model, tokenizer, sample, precision="int4", K=args.K, seed=args.seed + b
            )
            e_reid = efficacy_on(model, tokenizer, eval_items, reid_b.heads, 1.0, reid_b.steer_target)
            reid_deltas.append(e_reid - transferred)

            best_b = transferred
            for a in alphas:
                e = efficacy_on(model, tokenizer, eval_items, fp16_heads.heads, a, fp16_heads.steer_target)
                best_b = max(best_b, e)
            rescale_deltas.append(best_b - transferred)

    finally:
        unload_model(model)

    reid_mean, reid_lo, reid_hi = bootstrap_ci(reid_deltas)
    rescale_mean, rescale_lo, rescale_hi = bootstrap_ci(rescale_deltas)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_identified_heads(args.out_dir / f"heads_fp16_for_transfer_{stamp}.json", fp16_heads)

    report = {
        "transferred_efficacy": transferred,
        "rescale_only_best": {"alpha": best_alpha, "efficacy": best_rescale},
        "reid_32_efficacy": reid_eff,
        "negative_control_efficacy": neg_eff,
        "delta_reid_minus_transferred": reid_eff - transferred,
        "delta_rescale_minus_transferred": best_rescale - transferred,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_reid_delta_ci": {"mean": reid_mean, "lo": reid_lo, "hi": reid_hi},
        "bootstrap_rescale_delta_ci": {"mean": rescale_mean, "lo": rescale_lo, "hi": rescale_hi},
        "ci_overlap_unresolved": not (reid_hi < rescale_lo or rescale_hi < reid_lo),
    }
    out_path = args.out_dir / f"repair_fix1_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
