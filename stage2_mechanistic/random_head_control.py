#!/usr/bin/env python3
"""Random-K head control at the same alpha as identified heads."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-K JuICE control (1B)")
    p.add_argument("--heads", type=Path, required=True, help="Reference identified heads JSON (for K, alpha, exclude)")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "conflictqa_smoke.jsonl")
    p.add_argument("--n_items", type=int, default=50)
    p.add_argument("--apply_precision", choices=("fp16", "int4"), default="fp16")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--out_dir", type=Path, default=ROOT / "stage2_mechanistic" / "outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from common.conflictqa import load_conflict_items
    from common.juice import (
        IdentifiedHeads,
        apply_juice_steering,
        load_identified_heads,
        model_dims,
        random_head_control,
        steering_efficacy,
    )
    from common.models import load_model_and_tokenizer, unload_model

    ref = load_identified_heads(args.heads)
    items = load_conflict_items(args.data, n_items=args.n_items)
    alpha = args.alpha if args.alpha is not None else 1.0

    model, tokenizer = load_model_and_tokenizer(args.apply_precision)  # type: ignore[arg-type]
    n_layers, n_heads, _ = model_dims(model)
    exclude = list(ref.enlarge) + list(ref.suppress)
    rand_heads = random_head_control(n_layers, n_heads, ref.K, seed=args.seed, exclude=exclude)
    rand_ident = IdentifiedHeads(
        enlarge=rand_heads,
        suppress=[],
        scores_pos={h.key(): 0.0 for h in rand_heads},
        scores_neg={},
        precision=args.apply_precision,
        K=ref.K,
        steer_target=ref.steer_target,
        n_calibration=0,
        pos_alphas=[],
        neg_alphas=[],
    )
    try:
        results = apply_juice_steering(
            model,
            tokenizer,
            items,
            rand_ident,
            alpha_plus=alpha,
            alpha_minus=0.0,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        unload_model(model)

    eff = steering_efficacy(results, ref.steer_target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"random_k_{args.apply_precision}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "control": "random_k",
                "K": ref.K,
                "alpha": alpha,
                "seed": args.seed,
                "heads": [h.key() for h in rand_heads],
                "efficacy": eff,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Random-K efficacy={eff:.3f}; wrote {out_path}")


if __name__ == "__main__":
    main()
