"""Paired n=100 table plus H- specificity / DiD (same protocol as n=50)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

N, SEED = 100, 42
ROOT = Path("/kaggle/working")
LOCKED_RANDOM = [
    "L41_H11",
    "L7_H3",
    "L1_H9",
    "L17_H13",
    "L15_H13",
    "L14_H7",
    "L9_H1",
    "L6_H9",
    "L35_H9",
    "L5_H10",
]


def find_dir(name: str, marker: str) -> Path:
    hits = []
    for root in (ROOT, Path("/kaggle/input")):
        if root.exists():
            hits.extend(p.parent for p in root.rglob(marker) if p.is_file())
    hits.sort(
        key=lambda p: (
            0 if "working" in str(p).replace("\\", "/") else 1,
            0 if p.name == name else 1,
            len(str(p)),
        )
    )
    assert hits, f"Missing {name}/{marker}"
    print(name, "->", hits[0])
    return hits[0]


def rows(path: Path) -> dict:
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def fc(row) -> bool:
    return row["label"] == "context"


def rate(ids, tab) -> float:
    return sum(fc(tab[i]) for i in ids) / len(ids)


def pco(tab):
    return {k: sum(r["label"] == k for r in tab.values()) for k in ("parametric", "context", "other")}


def coherent(text: str) -> bool:
    import re

    t = (text or "").strip()
    if len(t) < 2 or t.count("://") >= 3:
        return False
    if re.fullmatch(r"[\W\d_]+", t):
        return False
    toks = t.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters >= max(2, int(0.3 * len(t)))


def coh(tab) -> float:
    return sum(coherent(r.get("text")) for r in tab.values()) / len(tab)


def ci(xs):
    lo, hi = np.percentile(xs, [2.5, 97.5])
    return [100 * float(lo), 100 * float(hi)]


def stat(point, draws):
    lo, hi = ci(draws)
    return {"delta_pp": 100 * point, "ci95_pp": [lo, hi], "excludes_zero": not (lo <= 0 <= hi)}


def main() -> None:
    dirs = {
        "fp16": find_dir("gemma_n100_fp16", "steer_parametric_k1_raw.json"),
        "int4": find_dir("gemma_n100_int4_headline", "random_context_k10.json"),
        "fp16_id_hminus": find_dir("gemma_n100_fp16_hminus", "suppress_context_k10.json"),
        "fp16_rand_hminus": find_dir("gemma_n100_fp16_random_hminus", "random_suppress_k10.json"),
        "int4_hminus": find_dir("gemma_n100_int4_hminus", "random_suppress_k10.json"),
    }
    fp16_rand = json.loads((dirs["fp16"] / "random_k10_heads.json").read_text(encoding="utf-8"))["heads"]
    int4_rand = json.loads((dirs["int4"] / "random_k10_heads.json").read_text(encoding="utf-8"))["heads"]
    hminus_rand = json.loads((dirs["fp16_rand_hminus"] / "random_k10_heads.json").read_text(encoding="utf-8"))["heads"]
    int4_hminus_rand = json.loads((dirs["int4_hminus"] / "random_k10_heads.json").read_text(encoding="utf-8"))["heads"]
    assert fp16_rand == LOCKED_RANDOM, fp16_rand
    assert int4_rand == LOCKED_RANDOM, int4_rand
    assert hminus_rand == LOCKED_RANDOM, hminus_rand
    assert int4_hminus_rand == LOCKED_RANDOM, int4_hminus_rand
    print("locked random K10 confirmed", LOCKED_RANDOM)

    fp, i4 = dirs["fp16"], dirs["int4"]
    tabs = {
        "fp16_baseline": rows(fp / "steer_baseline.json"),
        "fp16_context_k10": rows(fp / "steer_context_k10.json"),
        "fp16_parametric_k2_clean": rows(fp / "steer_parametric_k2_clean.json"),
        "fp16_raw_l27_h6": rows(fp / "steer_parametric_k1_raw.json"),
        "int4_baseline": rows(i4 / "steer_baseline.json"),
        "int4_context_k10": rows(i4 / "steer_context_k10.json"),
        "int4_random_k10": rows(i4 / "random_context_k10.json"),
        "int4_raw_l27_h6": rows(i4 / "steer_parametric_k1_raw.json"),
        "int4_random_k1": rows(i4 / "random_parametric_k1.json"),
        "fp16_hminus_id": rows(dirs["fp16_id_hminus"] / "suppress_context_k10.json"),
        "fp16_hminus_id_base": rows(dirs["fp16_id_hminus"] / "steer_baseline.json"),
        "fp16_hminus_rand": rows(dirs["fp16_rand_hminus"] / "random_suppress_k10.json"),
        "fp16_hminus_rand_base": rows(dirs["fp16_rand_hminus"] / "steer_baseline.json"),
        "int4_hminus_id": rows(dirs["int4_hminus"] / "suppress_context_k10.json"),
        "int4_hminus_rand": rows(dirs["int4_hminus"] / "random_suppress_k10.json"),
        "int4_hminus_base": rows(dirs["int4_hminus"] / "steer_baseline.json"),
    }
    ids = sorted(set.intersection(*(set(x) for x in tabs.values())))
    assert len(ids) == N, {k: len(v) for k, v in tabs.items()}
    rng = np.random.default_rng(SEED)
    draws = [[ids[j] for j in rng.integers(0, N, size=N)] for _ in range(1000)]

    def delta(a, b, d):
        return rate(d, tabs[a]) - rate(d, tabs[b])

    def boot(a, base, name):
        point = delta(a, base, ids)
        xs = [delta(a, base, d) for d in draws]
        return {
            "condition": name,
            "fc": rate(ids, tabs[a]),
            "baseline_fc": rate(ids, tabs[base]),
            "pco": pco(tabs[a]),
            "coherence": coh(tabs[a]),
            **stat(point, xs),
        }

    out = [
        boot("fp16_context_k10", "fp16_baseline", "FP16 context K10"),
        boot("int4_context_k10", "int4_baseline", "INT4 context K10 transfer"),
        boot("int4_random_k10", "int4_baseline", "INT4 random K10"),
        boot("fp16_parametric_k2_clean", "fp16_baseline", "FP16 clean parametric K2"),
        boot("fp16_raw_l27_h6", "fp16_baseline", "FP16 raw L27_H6"),
        boot("int4_raw_l27_h6", "int4_baseline", "INT4 raw L27_H6 transfer"),
        boot("int4_random_k1", "int4_baseline", "INT4 random K1"),
        boot("fp16_hminus_id", "fp16_hminus_id_base", "FP16 identified H-"),
        boot("fp16_hminus_rand", "fp16_hminus_rand_base", "FP16 random H-"),
        boot("int4_hminus_id", "int4_hminus_base", "INT4 identified H-"),
        boot("int4_hminus_rand", "int4_hminus_base", "INT4 random H-"),
    ]

    def d_id(d):
        return delta("fp16_hminus_id", "fp16_hminus_id_base", d)

    def d_rand(d):
        return delta("fp16_hminus_rand", "fp16_hminus_rand_base", d)

    def d4_id(d):
        return delta("int4_hminus_id", "int4_hminus_base", d)

    def d4_rand(d):
        return delta("int4_hminus_rand", "int4_hminus_base", d)

    point = {
        "delta_fp16_id": d_id(ids),
        "delta_fp16_random": d_rand(ids),
        "delta_int4_id": d4_id(ids),
        "delta_int4_random": d4_rand(ids),
    }
    point["specificity_fp16"] = point["delta_fp16_id"] - point["delta_fp16_random"]
    point["specificity_int4"] = point["delta_int4_id"] - point["delta_int4_random"]
    point["difference_in_differences"] = point["specificity_int4"] - point["specificity_fp16"]
    samples = {k: [] for k in point}
    for d in draws:
        a, b, c, e = d_id(d), d_rand(d), d4_id(d), d4_rand(d)
        spec16 = a - b
        spec4 = c - e
        samples["delta_fp16_id"].append(a)
        samples["delta_fp16_random"].append(b)
        samples["delta_int4_id"].append(c)
        samples["delta_int4_random"].append(e)
        samples["specificity_fp16"].append(spec16)
        samples["specificity_int4"].append(spec4)
        samples["difference_in_differences"].append(spec4 - spec16)

    def summarize(key):
        xs = samples[key]
        p = point[key]
        lo, hi = float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))
        return {
            "point": p,
            "point_pp": 100 * p,
            "ci95": [lo, hi],
            "ci95_pp": [100 * lo, 100 * hi],
            "excludes_zero": not (lo <= 0 <= hi),
            "n_draws_lt0": sum(x < 0 for x in xs),
            "n_draws_eq0": sum(x == 0 for x in xs),
            "n_draws_gt0": sum(x > 0 for x in xs),
        }

    estimates = {k: summarize(k) for k in point}
    spec16 = {
        "delta_pp": estimates["specificity_fp16"]["point_pp"],
        "ci95_pp": estimates["specificity_fp16"]["ci95_pp"],
        "excludes_zero": estimates["specificity_fp16"]["excludes_zero"],
    }
    spec4 = {
        "delta_pp": estimates["specificity_int4"]["point_pp"],
        "ci95_pp": estimates["specificity_int4"]["ci95_pp"],
        "excludes_zero": estimates["specificity_int4"]["excludes_zero"],
    }
    did = {
        "delta_pp": estimates["difference_in_differences"]["point_pp"],
        "ci95_pp": estimates["difference_in_differences"]["ci95_pp"],
        "excludes_zero": estimates["difference_in_differences"]["excludes_zero"],
    }
    payload = {
        "n": N,
        "n_boot": 1000,
        "seed": SEED,
        "ids": ids,
        "cells": out,
        "specificity_fp16": spec16,
        "specificity_int4": spec4,
        "difference_in_differences": did,
        "locked_random_k10": LOCKED_RANDOM,
    }
    (ROOT / "gemma_n100_raw_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    did_payload = {
        "n": N,
        "n_boot": 1000,
        "seed": SEED,
        "ids": ids,
        "fp16_baseline_itemwise_agree": sum(
            tabs["fp16_hminus_id_base"][i]["label"] == tabs["fp16_hminus_rand_base"][i]["label"]
            for i in ids
        ),
        "note": (
            "Paired item-level bootstrap. Each draw resamples the 100 eval IDs "
            "with replacement. Deltas use each cell's own baseline (FP16 identified "
            "and FP16 random are separate sessions). specificity = delta_id − "
            "delta_random. DiD = specificity_int4 − specificity_fp16. "
            "Percentile 95% CI. Same protocol as n=50."
        ),
        "locked_random_k10": LOCKED_RANDOM,
        "estimates": estimates,
    }
    (ROOT / "gemma_n100_hminus_specificity_did.json").write_text(
        json.dumps(did_payload, indent=2), encoding="utf-8"
    )
    with (ROOT / "gemma_n100_raw_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["condition", "fc", "delta_pp", "ci95_pp", "excludes_zero", "pco", "coherence"]
        )
        w.writeheader()
        for x in out:
            w.writerow(x)
    print(json.dumps({"cells": out, "estimates": estimates}, indent=2))
    print("Wrote gemma_n100_raw_results.json, .csv, and gemma_n100_hminus_specificity_did.json")


if __name__ == "__main__":
    main()
