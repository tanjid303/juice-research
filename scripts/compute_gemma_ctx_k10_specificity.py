"""Compute paired specificity contrast and DiD for Gemma context K10 enlarge.

Protocol matches scripts/rescore_gemma_hminus_question_aware.py:
- n=100 items
- question-aware rescore: label_conflict_response(..., question=item.question)
- follows_context: label == 'context'
- delta = steered - baseline (own session baseline)
- specificity = delta_identified - delta_random (paired on identical bootstrap draws)
- DiD = spec_INT4 - spec_FP16 (paired on identical bootstrap draws)
- 1000 draws, seed 42, percentile 95% CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\tt")
sys.path.insert(0, str(ROOT))
from common.scoring import label_conflict_response

DATA = ROOT / "data" / "conflictqa_pilot250_popular.jsonl"
gold = {}
for line in DATA.read_text(encoding="utf-8").splitlines():
    if line.strip():
        o = json.loads(line)
        gold[o["id"]] = o

def load_and_rescore(path: Path) -> dict[str, bool]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    fc = {}
    for r in rows:
        gid = r["id"]
        g = gold[gid]
        lbl = label_conflict_response(r["text"], g["parametric_answer"], g["context_answer"], question=g["question"])
        fc[gid] = (lbl == "context")
    return fc

def main():
    fp16_base_p = ROOT / "last-one-g" / "gemma_n100_fp16" / "steer_baseline.json"
    fp16_id_p = ROOT / "last-one-g" / "gemma_n100_fp16" / "steer_context_k10.json"
    fp16_rand_p = ROOT / "last-one-g" / "gemma_n100_fp16" / "random_context_k10.json"

    int4_base_p = ROOT / "last-one-g" / "gemma_n100_int4_headline" / "steer_baseline.json"
    int4_id_p = ROOT / "last-one-g" / "gemma_n100_int4_headline" / "steer_context_k10.json"
    int4_rand_p = ROOT / "last-one-g" / "gemma_n100_int4_headline" / "random_context_k10.json"

    fc_fp16_base = load_and_rescore(fp16_base_p)
    fc_fp16_id = load_and_rescore(fp16_id_p)
    fc_fp16_rand = load_and_rescore(fp16_rand_p)

    fc_int4_base = load_and_rescore(int4_base_p)
    fc_int4_id = load_and_rescore(int4_id_p)
    fc_int4_rand = load_and_rescore(int4_rand_p)

    ids = sorted(fc_fp16_base.keys())
    n = len(ids)
    print(f"Loaded {n} items.")

    N_BOOT = 1000
    SEED = 42
    rng = np.random.default_rng(SEED)

    # Point estimates
    d_fp16_id = (sum(fc_fp16_id[i] for i in ids) - sum(fc_fp16_base[i] for i in ids)) / n
    d_fp16_rand = (sum(fc_fp16_rand[i] for i in ids) - sum(fc_fp16_base[i] for i in ids)) / n
    spec_fp16 = d_fp16_id - d_fp16_rand

    d_int4_id = (sum(fc_int4_id[i] for i in ids) - sum(fc_int4_base[i] for i in ids)) / n
    d_int4_rand = (sum(fc_int4_rand[i] for i in ids) - sum(fc_int4_base[i] for i in ids)) / n
    spec_int4 = d_int4_id - d_int4_rand

    did = spec_int4 - spec_fp16

    draws_fp16_id = []
    draws_fp16_rand = []
    draws_spec_fp16 = []

    draws_int4_id = []
    draws_int4_rand = []
    draws_spec_int4 = []

    draws_did = []

    for _ in range(N_BOOT):
        draw = [ids[j] for j in rng.integers(0, n, size=n)]
        df16_id = (sum(fc_fp16_id[i] for i in draw) - sum(fc_fp16_base[i] for i in draw)) / n
        df16_rand = (sum(fc_fp16_rand[i] for i in draw) - sum(fc_fp16_base[i] for i in draw)) / n
        s16 = df16_id - df16_rand

        di4_id = (sum(fc_int4_id[i] for i in draw) - sum(fc_int4_base[i] for i in draw)) / n
        di4_rand = (sum(fc_int4_rand[i] for i in draw) - sum(fc_int4_base[i] for i in draw)) / n
        s4 = di4_id - di4_rand

        d_id = s4 - s16

        draws_fp16_id.append(df16_id)
        draws_fp16_rand.append(df16_rand)
        draws_spec_fp16.append(s16)

        draws_int4_id.append(di4_id)
        draws_int4_rand.append(di4_rand)
        draws_spec_int4.append(s4)
        draws_did.append(d_id)

    def pack(name, pt, draws):
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        excl = not (lo <= 0 <= hi)
        print(f"{name:30s}: {100*pt:+5.1f} [{100*lo:+5.1f}, {100*hi:+5.1f}]  (excludes 0: {excl})")
        return {
            "name": name,
            "point_pp": round(100 * pt, 1),
            "ci95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
            "excludes_zero": excl,
        }

    res = {
        "fp16_id": pack("FP16 ID delta", d_fp16_id, draws_fp16_id),
        "fp16_rand": pack("FP16 RAND delta", d_fp16_rand, draws_fp16_rand),
        "fp16_specificity": pack("FP16 Specificity (ID - RAND)", spec_fp16, draws_spec_fp16),
        "int4_id": pack("INT4 ID delta", d_int4_id, draws_int4_id),
        "int4_rand": pack("INT4 RAND delta", d_int4_rand, draws_int4_rand),
        "int4_specificity": pack("INT4 Specificity (ID - RAND)", spec_int4, draws_spec_int4),
        "did": pack("DiD (spec_INT4 - spec_FP16)", did, draws_did),
    }

    out_p = ROOT / "scripts" / "gemma_ctx_k10_specificity_results.json"
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_p}")

if __name__ == "__main__":
    main()
