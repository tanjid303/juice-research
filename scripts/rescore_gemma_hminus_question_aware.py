"""Rescore Gemma H- rows with question-aware label_conflict_response.

Does not rewrite original run JSONs. Writes a standalone report.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\tt")
sys.path.insert(0, str(ROOT))
from common.scoring import label_conflict_response  # noqa: E402

N_BOOT = 1000
SEED = 42
DATA = ROOT / "data" / "conflictqa_pilot250_popular.jsonl"
OUT = ROOT / "last-one-g" / "gemma_hminus_question_aware_rescore.json"

N100 = {
    "fp16_id_base": ROOT / "last-one-g" / "gemma_n100_fp16_hminus" / "steer_baseline.json",
    "fp16_id": ROOT / "last-one-g" / "gemma_n100_fp16_hminus" / "suppress_context_k10.json",
    "fp16_rand_base": ROOT / "last-one-g" / "gemma_n100_fp16_random_hminus" / "steer_baseline.json",
    "fp16_rand": ROOT / "last-one-g" / "gemma_n100_fp16_random_hminus" / "random_suppress_k10.json",
    "int4_base": ROOT / "last-one-g" / "gemma_n100_int4_hminus" / "steer_baseline.json",
    "int4_id": ROOT / "last-one-g" / "gemma_n100_int4_hminus" / "suppress_context_k10.json",
    "int4_rand": ROOT / "last-one-g" / "gemma_n100_int4_hminus" / "random_suppress_k10.json",
}
N50 = {
    "fp16_id_base": ROOT / "task-2-res" / "juice_stage2_gemma2_fp16_context_suppress" / "steer_baseline.json",
    "fp16_id": ROOT / "task-2-res" / "juice_stage2_gemma2_fp16_context_suppress" / "suppress_context_k10.json",
    "fp16_rand_base": ROOT / "res-fp16-test-gemma" / "juice_stage2_gemma2_fp16_random_k10_suppress" / "steer_baseline.json",
    "fp16_rand": ROOT / "res-fp16-test-gemma" / "juice_stage2_gemma2_fp16_random_k10_suppress" / "random_suppress_k10.json",
    "int4_base": ROOT / "gemma-int4-kicu" / "juice_stage2_gemma2_int4_context_suppress" / "steer_baseline.json",
    "int4_id": ROOT / "gemma-int4-kicu" / "juice_stage2_gemma2_int4_context_suppress" / "suppress_context_k10.json",
    "int4_rand": ROOT / "gemma-int4-kicu" / "juice_stage2_gemma2_int4_context_suppress" / "random_suppress_k10.json",
}


def load_gold() -> dict:
    out = {}
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            out[o["id"]] = o
    return out


def load_rows(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def rescore_row(row: dict, gold: dict) -> str:
    return label_conflict_response(
        row["text"],
        gold["parametric_answer"],
        gold["context_answer"],
        question=gold["question"],
    )


def pco(labels: list[str]) -> dict:
    c = Counter(labels)
    return {
        "parametric": int(c.get("parametric", 0)),
        "context": int(c.get("context", 0)),
        "rejects_context": int(c.get("rejects_context", 0)),
        "other": int(c.get("other", 0)),
        "pco": f"{c.get('parametric', 0)}/{c.get('context', 0)}/{c.get('other', 0)}",
    }


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{100 * lo:+.1f}, {100 * hi:+.1f}]"


def boot_delta(ids: list[str], base_fc: dict[str, bool], ste_fc: dict[str, bool]) -> dict:
    n = len(ids)
    rng = np.random.default_rng(SEED)
    point = sum(ste_fc[i] for i in ids) / n - sum(base_fc[i] for i in ids) / n
    draws = []
    for _ in range(N_BOOT):
        draw = [ids[j] for j in rng.integers(0, n, size=n)]
        draws.append(sum(ste_fc[i] for i in draw) / n - sum(base_fc[i] for i in draw) / n)
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {
        "fc_base": sum(base_fc[i] for i in ids) / n,
        "fc_steered": sum(ste_fc[i] for i in ids) / n,
        "delta": point,
        "delta_pp": 100 * point,
        "ci95": [lo, hi],
        "ci95_pp": [100 * lo, 100 * hi],
        "excludes_zero": not (lo <= 0.0 <= hi),
        "n_draws_lt0": sum(x < 0 for x in draws),
        "n_draws_eq0": sum(x == 0 for x in draws),
        "n_draws_gt0": sum(x > 0 for x in draws),
        "_draws": draws,
        "_point": point,
    }


def analyze(paths: dict, gold: dict, expected_n: int, tag: str) -> dict:
    tables_raw = {k: load_rows(p) for k, p in paths.items()}
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    id_sets = [set(v) for v in tables_raw.values()]
    ids = sorted(set.intersection(*id_sets))
    assert len(ids) == expected_n, {k: len(v) for k, v in tables_raw.items()}
    for i in ids:
        assert i in gold, i

    labels = {}
    n_changed = {}
    for k, tab in tables_raw.items():
        new = {}
        changed = 0
        for i in ids:
            lab = rescore_row(tab[i], gold[i])
            old = tab[i]["label"]
            if lab != old:
                changed += 1
            new[i] = lab
        labels[k] = new
        n_changed[k] = changed

    def fc_map(labmap):
        return {i: labmap[i] == "context" for i in ids}

    conditions = {
        "FP16 identified H-": ("fp16_id", "fp16_id_base"),
        "FP16 random H-": ("fp16_rand", "fp16_rand_base"),
        "INT4 identified H-": ("int4_id", "int4_base"),
        "INT4 random H-": ("int4_rand", "int4_base"),
    }
    cells = {}
    for name, (ste_k, base_k) in conditions.items():
        ste_labs = [labels[ste_k][i] for i in ids]
        base_labs = [labels[base_k][i] for i in ids]
        orig_ste = [tables_raw[ste_k][i]["label"] for i in ids]
        orig_base = [tables_raw[base_k][i]["label"] for i in ids]
        orig_delta = (sum(x == "context" for x in orig_ste) - sum(x == "context" for x in orig_base)) / len(ids)
        boot = boot_delta(ids, fc_map(labels[base_k]), fc_map(labels[ste_k]))
        cells[name] = {
            "n": expected_n,
            "rescored": {
                "fc": boot["fc_steered"],
                "baseline_fc": boot["fc_base"],
                "delta_pp": boot["delta_pp"],
                "ci95_pp": boot["ci95_pp"],
                "excludes_zero": boot["excludes_zero"],
                "pco_steered": pco(ste_labs),
                "pco_baseline": pco(base_labs),
            },
            "original": {
                "fc": sum(x == "context" for x in orig_ste) / len(ids),
                "baseline_fc": sum(x == "context" for x in orig_base) / len(ids),
                "delta_pp": 100 * orig_delta,
                "pco_steered": pco(orig_ste),
            },
            "n_labels_changed_steered": n_changed[ste_k],
            "n_labels_changed_baseline": n_changed[base_k],
            "detection_flip": None,
            "_draws": boot["_draws"],
            "_point": boot["_point"],
        }

    # Original detection from published n=100 / n=50 summaries (hardcoded for comparison).
    orig_detect = {
        100: {
            "FP16 identified H-": True,
            "FP16 random H-": False,
            "INT4 identified H-": True,
            "INT4 random H-": True,
        },
        50: {
            "FP16 identified H-": True,
            "FP16 random H-": False,
            "INT4 identified H-": True,
            "INT4 random H-": True,
        },
    }
    for name, cell in cells.items():
        was = orig_detect[expected_n][name]
        now = cell["rescored"]["excludes_zero"]
        cell["detection_flip"] = (was, now, was != now)

    def spec_did():
        a = cells["FP16 identified H-"]["_point"]
        b = cells["FP16 random H-"]["_point"]
        c = cells["INT4 identified H-"]["_point"]
        d = cells["INT4 random H-"]["_point"]
        da, db, dc, dd = (
            cells["FP16 identified H-"]["_draws"],
            cells["FP16 random H-"]["_draws"],
            cells["INT4 identified H-"]["_draws"],
            cells["INT4 random H-"]["_draws"],
        )
        spec16 = a - b
        spec4 = c - d
        did = spec4 - spec16
        spec16_d = [x - y for x, y in zip(da, db)]
        spec4_d = [x - y for x, y in zip(dc, dd)]
        did_d = [x - y for x, y in zip(spec4_d, spec16_d)]

        def pack(point, xs):
            lo, hi = float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))
            return {
                "point": point,
                "point_pp": 100 * point,
                "ci95_pp": [100 * lo, 100 * hi],
                "excludes_zero": not (lo <= 0.0 <= hi),
                "n_draws_lt0": sum(x < 0 for x in xs),
                "n_draws_eq0": sum(x == 0 for x in xs),
                "n_draws_gt0": sum(x > 0 for x in xs),
            }

        return {
            "specificity_fp16": pack(spec16, spec16_d),
            "specificity_int4": pack(spec4, spec4_d),
            "difference_in_differences": pack(did, did_d),
        }

    summary = spec_did()
    # drop draws from cells for json
    cells_out = {}
    for name, cell in cells.items():
        cells_out[name] = {k: v for k, v in cell.items() if not k.startswith("_")}

    print(f"\n======== {tag} n={expected_n}  question-aware rescore ========")
    print(f"{'condition':24s} {'orig d':>10s} {'resc d':>18s} {'excl0 orig->new':>16s}  P/C/O")
    for name, cell in cells.items():
        r = cell["rescored"]
        o = cell["original"]
        was, now, flipped = cell["detection_flip"]
        det = f"{'Y' if was else 'n'}→{'Y' if now else 'n'}{' *' if flipped else ''}"
        ci = fmt_ci(r["ci95_pp"][0] / 100, r["ci95_pp"][1] / 100)
        print(
            f"{name:24s} {o['delta_pp']:+6.1f}  {r['delta_pp']:+6.1f} {ci:>14s}  {det:8s}  "
            f"{r['pco_steered']['pco']}  (base {r['pco_baseline']['pco']}; "
            f"labels d steered {cell['n_labels_changed_steered']}, base {cell['n_labels_changed_baseline']})"
        )
    print("--- derived ---")
    for k in ("specificity_fp16", "specificity_int4", "difference_in_differences"):
        s = summary[k]
        print(
            f"{k:28s} {s['point_pp']:+6.1f}  {fmt_ci(s['ci95_pp'][0]/100, s['ci95_pp'][1]/100):>14s}  "
            f"excl0={s['excludes_zero']}"
        )
    return {
        "n": expected_n,
        "n_boot": N_BOOT,
        "seed": SEED,
        "n_ids": len(ids),
        "note": (
            "Relabel with label_conflict_response(..., question=item.question). "
            "Deltas use each cell's own rescored baseline (n=50 protocol). "
            "specificity = delta_id − delta_random. DiD = spec_INT4 − spec_FP16."
        ),
        "cells": cells_out,
        "derived": summary,
        "n_labels_changed": n_changed,
    }


def main() -> None:
    gold = load_gold()
    n100 = analyze(N100, gold, 100, "n=100 last-one-g")
    n50 = analyze(N50, gold, 50, "n=50 original H-")
    payload = {
        "matcher": "label_conflict_response(..., question=gold.question)",
        "n100": n100,
        "n50": n50,
        "published_original": {
            "n100": {
                "FP16 identified": "-13 [-21, -7] detected",
                "FP16 random": "-6 [-13, +1] null",
                "INT4 identified": "-8 [-15, -1] detected",
                "INT4 random": "-13 [-20, -6] detected",
                "specificity_fp16": "-7 [-15, +1] null",
                "specificity_int4": "+5 [-3, +14] null",
                "DiD": "+12 [+2, +23] detected",
            },
            "n50": {
                "FP16 identified": "-12 [-24, -2] detected",
                "FP16 random": "-6 [-16, +4] null",
                "INT4 identified": "-8 [-16, -2] detected",
                "INT4 random": "-12 [-24, -2] detected",
                "specificity_fp16": "-6 [-18, +6] null",
                "specificity_int4": "+4 [-8, +16] null",
                "DiD": "+10 [-4, +24] null",
            },
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
