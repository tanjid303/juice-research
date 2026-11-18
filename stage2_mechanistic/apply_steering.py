#!/usr/bin/env python3
"""
Apply JuICE steering — 2×2 identify×apply cells for 1B pilot.

Cells (run via --identify_precision / --apply_precision):
  FP16→FP16 (reproduction)
  FP16→INT4 (transfer)
  INT4→INT4 (native)
  INT4→FP16 (reverse transfer)

Usage:
  python stage2_mechanistic/identify_heads.py --precision fp16 --out stage2_mechanistic/outputs/heads_fp16.json
  python stage2_mechanistic/apply_steering.py --heads stage2_mechanistic/outputs/heads_fp16.json --apply_precision int4
"""

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
    p = argparse.ArgumentParser(description="JuICE 2x2 apply (1B)")
    p.add_argument("--heads", type=Path, required=True, help="JSON from identify_heads.py")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "conflictqa_smoke.jsonl")
    p.add_argument("--n_items", type=int, default=50)
    p.add_argument("--apply_precision", choices=("fp16", "int4"), default="fp16")
    p.add_argument("--alpha_plus", type=float, default=5.0)
    p.add_argument("--alpha_minus", type=float, default=2.0)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--out_dir", type=Path, default=ROOT / "stage2_mechanistic" / "outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from common.conflictqa import load_conflict_items
    from common.juice import apply_juice_steering, load_identified_heads, steering_efficacy
    from common.models import load_model_and_tokenizer, unload_model

    identified = load_identified_heads(args.heads)
    items = load_conflict_items(args.data, n_items=args.n_items)

    model, tokenizer = load_model_and_tokenizer(args.apply_precision)  # type: ignore[arg-type]
    try:
        results = apply_juice_steering(
            model,
            tokenizer,
            items,
            identified,
            alpha_plus=args.alpha_plus,
            alpha_minus=args.alpha_minus,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        unload_model(model)

    eff = steering_efficacy(results, identified.steer_target)
    cell = f"id_{identified.precision}_apply_{args.apply_precision}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"steer_{cell}_{stamp}.json"
    payload = {
        "cell": cell,
        "identify_precision": identified.precision,
        "apply_precision": args.apply_precision,
        "steer_target": identified.steer_target,
        "n_items": len(results),
        "efficacy": eff,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Cell {cell}: efficacy={eff:.3f} ({identified.steer_target})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
