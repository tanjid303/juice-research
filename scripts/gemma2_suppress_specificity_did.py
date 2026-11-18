"""Paired bootstrap DiD: Gemma-2 ctx-K10 H− specificity, FP16 vs INT4.

Resamples the same 50 eval IDs with replacement (n_boot=1000, seed=42).
Does not compare overlapping CIs; the interaction (specificity_int4 −
specificity_fp16) is the inferential test.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"d:\tt")
OUT = ROOT / "juice_stage2_gemma2_suppress_did"
N_BOOT = 1000
SEED = 42

PATHS = {
    "fp16_baseline_id": ROOT
    / "task-2-res"
    / "juice_stage2_gemma2_fp16_context_suppress"
    / "steer_baseline.json",
    "fp16_identified": ROOT
    / "task-2-res"
    / "juice_stage2_gemma2_fp16_context_suppress"
    / "suppress_context_k10.json",
    "fp16_baseline_rand": ROOT
    / "res-fp16-test-gemma"
    / "juice_stage2_gemma2_fp16_random_k10_suppress"
    / "steer_baseline.json",
    "fp16_random": ROOT
    / "res-fp16-test-gemma"
    / "juice_stage2_gemma2_fp16_random_k10_suppress"
    / "random_suppress_k10.json",
    "int4_baseline": ROOT
    / "gemma-int4-kicu"
    / "juice_stage2_gemma2_int4_context_suppress"
    / "steer_baseline.json",
    "int4_identified": ROOT
    / "gemma-int4-kicu"
    / "juice_stage2_gemma2_int4_context_suppress"
    / "suppress_context_k10.json",
    "int4_random": ROOT
    / "gemma-int4-kicu"
    / "juice_stage2_gemma2_int4_context_suppress"
    / "random_suppress_k10.json",
}


def load_fc(path: Path) -> dict[str, bool]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        out[r["id"]] = bool(r.get("follows_context") or r.get("label") == "context")
    return out


def rate(ids: list[str], table: dict[str, bool]) -> float:
    return sum(table[i] for i in ids) / len(ids)


def summarize(xs: list[float], point: float) -> dict:
    lo, hi = float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))
    return {
        "point": point,
        "point_pp": 100.0 * point,
        "ci95": [lo, hi],
        "ci95_pp": [100.0 * lo, 100.0 * hi],
        "excludes_zero": not (lo <= 0.0 <= hi),
        "n_draws_lt0": sum(1 for x in xs if x < 0),
        "n_draws_eq0": sum(1 for x in xs if x == 0),
        "n_draws_gt0": sum(1 for x in xs if x > 0),
    }


def main() -> None:
    tables = {k: load_fc(p) for k, p in PATHS.items()}
    id_sets = {k: set(v) for k, v in tables.items()}
    ids = sorted(set.intersection(*id_sets.values()))
    assert len(ids) == 50, {k: len(v) for k, v in id_sets.items()}

    fp16_base_agree = sum(tables["fp16_baseline_id"][i] == tables["fp16_baseline_rand"][i] for i in ids)
    print("n_aligned", len(ids), "fp16_baseline_itemwise_agree", fp16_base_agree)

    def d_id(id_list: list[str]) -> float:
        return rate(id_list, tables["fp16_identified"]) - rate(id_list, tables["fp16_baseline_id"])

    def d_rand(id_list: list[str]) -> float:
        return rate(id_list, tables["fp16_random"]) - rate(id_list, tables["fp16_baseline_rand"])

    def d4_id(id_list: list[str]) -> float:
        return rate(id_list, tables["int4_identified"]) - rate(id_list, tables["int4_baseline"])

    def d4_rand(id_list: list[str]) -> float:
        return rate(id_list, tables["int4_random"]) - rate(id_list, tables["int4_baseline"])

    point = {
        "delta_fp16_id": d_id(ids),
        "delta_fp16_random": d_rand(ids),
        "delta_int4_id": d4_id(ids),
        "delta_int4_random": d4_rand(ids),
    }
    point["specificity_fp16"] = point["delta_fp16_id"] - point["delta_fp16_random"]
    point["specificity_int4"] = point["delta_int4_id"] - point["delta_int4_random"]
    point["difference_in_differences"] = point["specificity_int4"] - point["specificity_fp16"]

    rng = np.random.default_rng(SEED)
    samples = {k: [] for k in point}
    for _ in range(N_BOOT):
        draw = [ids[i] for i in rng.integers(0, len(ids), size=len(ids))]
        a = d_id(draw)
        b = d_rand(draw)
        c = d4_id(draw)
        d = d4_rand(draw)
        spec16 = a - b
        spec4 = c - d
        samples["delta_fp16_id"].append(a)
        samples["delta_fp16_random"].append(b)
        samples["delta_int4_id"].append(c)
        samples["delta_int4_random"].append(d)
        samples["specificity_fp16"].append(spec16)
        samples["specificity_int4"].append(spec4)
        samples["difference_in_differences"].append(spec4 - spec16)

    estimates = {k: summarize(samples[k], point[k]) for k in point}

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": len(ids),
        "n_boot": N_BOOT,
        "seed": SEED,
        "ids": ids,
        "fp16_baseline_itemwise_agree": fp16_base_agree,
        "note": (
            "Paired item-level bootstrap. Each draw resamples the 50 eval IDs "
            "with replacement. Deltas use each cell's own baseline (FP16 identified "
            "and FP16 random are separate sessions). specificity = delta_id − "
            "delta_random. DiD = specificity_int4 − specificity_fp16. "
            "Percentile 95% CI. Do not infer from overlapping CIs."
        ),
        "sources": {k: str(p.relative_to(ROOT).as_posix()) for k, p in PATHS.items()},
        "estimates": estimates,
    }
    (OUT / "gemma2_suppress_specificity_did.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (OUT / "gemma2_suppress_specificity_did_samples.json").write_text(
        json.dumps({"n_boot": N_BOOT, "seed": SEED, "samples": samples}, indent=2),
        encoding="utf-8",
    )
    csv_path = OUT / "gemma2_suppress_specificity_did_samples.csv"
    keys = list(samples)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["draw", *keys])
        w.writeheader()
        for i in range(N_BOOT):
            row = {"draw": i}
            row.update({k: samples[k][i] for k in keys})
            w.writerow(row)

    print(json.dumps(estimates, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
