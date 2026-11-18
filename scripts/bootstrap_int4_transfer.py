"""Paired item bootstrap of INT4 (and FP16) transfer deltas. n=50, 1000 draws."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"d:\tt")
import sys

sys.path.insert(0, str(ROOT))

from common.stats_analysis import bootstrap_paired_follow_delta  # noqa: E402

INT4 = ROOT / "int_4_result"
FP16 = ROOT / "juice_stage2_base_directional_cell3"
OUT = INT4 / "int4_transfer_bootstrap.json"

# INT4 downloads sometimes get " (1)" suffixes from the browser.
INT4_FILES = {
    "baseline": "steer_baseline (1).json",
    "steer_context_k10": "steer_context_k10 (1).json",
    "steer_context_k5": "steer_context_k5 (1).json",
    "steer_parametric_k2_clean": "steer_parametric_k2_clean.json",
    "steer_parametric_k2": "steer_parametric_k2 (1).json",
    "steer_parametric_k3": "steer_parametric_k3 (1).json",
}
FP16_FILES = {
    "baseline": "steer_baseline.json",
    "steer_context_k10": "steer_context_k10.json",
    "steer_context_k5": "steer_context_k5.json",
    "steer_parametric_k2_clean": "steer_parametric_k2_clean.json",
    "steer_parametric_k2": "steer_parametric_k2.json",
    "steer_parametric_k3": "steer_parametric_k3.json",
}

PRIMARY = [
    "steer_context_k10",
    "steer_context_k5",
    "steer_parametric_k2_clean",
    "steer_parametric_k3",
]


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(base_dir: Path, files: dict[str, str], precision: str) -> dict:
    baseline = load_rows(base_dir / files["baseline"])
    out: dict = {"precision": precision, "n_baseline": len(baseline)}
    conditions = {}
    for tag in files:
        if tag == "baseline":
            continue
        rows = load_rows(base_dir / files[tag])
        boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=1000, seed=42)
        conditions[tag] = boot
        d = boot["delta"]
        print(
            f"{precision:5s} {tag:32s}  d={d['point_pp']:+6.1f} pp  "
            f"CI95=[{d['ci95_pp'][0]:+6.1f}, {d['ci95_pp'][1]:+6.1f}]  "
            f"excludes0={d['excludes_zero']}"
        )
    out["conditions"] = conditions
    return out


def main() -> None:
    print("=== paired bootstrap n=50, 1000 draws, seed=42 ===")
    int4 = run_one(INT4, INT4_FILES, "int4")
    print()
    fp16 = run_one(FP16, FP16_FILES, "fp16")
    payload = {
        "n_boot": 1000,
        "seed": 42,
        "note": (
            "Resample the 50 eval items with replacement. Delta = steered "
            "follows_context − baseline follows_context, paired on item id."
        ),
        "primary_conditions": PRIMARY,
        "int4": int4,
        "fp16": fp16,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nwrote", OUT)

    print("\n=== primary read (INT4) ===")
    for tag in PRIMARY:
        d = int4["conditions"][tag]["delta"]
        fp = fp16["conditions"][tag]["delta"]
        print(
            f"  {tag}: INT4 {d['point_pp']:+.1f} [{d['ci95_pp'][0]:+.1f}, {d['ci95_pp'][1]:+.1f}] "
            f"{'excl0' if d['excludes_zero'] else 'includes 0'}  |  "
            f"FP16 {fp['point_pp']:+.1f} [{fp['ci95_pp'][0]:+.1f}, {fp['ci95_pp'][1]:+.1f}] "
            f"{'excl0' if fp['excludes_zero'] else 'includes 0'}"
        )


if __name__ == "__main__":
    main()
