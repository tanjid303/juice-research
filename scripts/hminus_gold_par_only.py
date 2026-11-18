"""Gold-PAR-only Δ for Gemma identified H− (same convention as Llama K2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\tt")
sys.path.insert(0, str(ROOT))
from common.scoring import label_conflict_response, _mentions_answer

N_BOOT, SEED = 1000, 42

CELLS = [
    (
        "n100_fp16",
        ROOT / "last-one-g" / "gemma_n100_fp16_hminus" / "steer_baseline.json",
        ROOT / "last-one-g" / "gemma_n100_fp16_hminus" / "suppress_context_k10.json",
    ),
    (
        "n100_int4",
        ROOT / "last-one-g" / "gemma_n100_int4_hminus" / "steer_baseline.json",
        ROOT / "last-one-g" / "gemma_n100_int4_hminus" / "suppress_context_k10.json",
    ),
    (
        "n50_fp16",
        ROOT / "task-2-res" / "juice_stage2_gemma2_fp16_context_suppress" / "steer_baseline.json",
        ROOT / "task-2-res" / "juice_stage2_gemma2_fp16_context_suppress" / "suppress_context_k10.json",
    ),
    (
        "n50_int4",
        ROOT / "gemma-int4-kicu" / "juice_stage2_gemma2_int4_context_suppress" / "steer_baseline.json",
        ROOT / "gemma-int4-kicu" / "juice_stage2_gemma2_int4_context_suppress" / "suppress_context_k10.json",
    ),
]


def gold_map() -> dict:
    out = {}
    for line in (ROOT / "data" / "conflictqa_pilot250_popular.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            o = json.loads(line)
            out[o["id"]] = o
    return out


def load_rows(path: Path) -> dict:
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def qa(text: str, g: dict) -> str:
    return label_conflict_response(
        text, g["parametric_answer"], g["context_answer"], question=g["question"]
    )


def fmt_delta(pp: float, lo: float, hi: float, excl: bool) -> str:
    call = "detected" if excl else "null"

    def ppv(x: float) -> str:
        v = round(x)
        if abs(x - v) < 1e-6:
            return f"{v:+.0f}"
        return f"{x:+.1f}"

    v = round(pp)
    s = f"{v:+.0f}" if abs(pp - v) < 1e-6 else f"{pp:+.1f}"
    return f"{s} [{ppv(lo)}, {ppv(hi)}] {call}"


def boot(ids, base_fc, ste_fc):
    n = len(ids)
    point = sum(ste_fc[i] for i in ids) / n - sum(base_fc[i] for i in ids) / n
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        draw = [ids[j] for j in rng.integers(0, n, size=n)]
        draws.append(
            sum(ste_fc[i] for i in draw) / n - sum(base_fc[i] for i in draw) / n
        )
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    excl = not (lo <= 0.0 <= hi)
    return {
        "delta_pp": 100 * point,
        "ci95_pp": [100 * lo, 100 * hi],
        "excludes_zero": excl,
        "fmt": fmt_delta(100 * point, 100 * lo, 100 * hi, excl),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    gold = gold_map()
    out = {}
    for name, bpath, spath in CELLS:
        base, ste = load_rows(bpath), load_rows(spath)
        ids = sorted(set(base) & set(ste))
        qa_b, qa_s, cons_s = {}, {}, {}
        n_par, n_leave = 0, 0
        for i in ids:
            g = gold[i]
            nlb, nls = qa(base[i]["text"], g), qa(ste[i]["text"], g)
            qa_b[i] = nlb == "context"
            qa_s[i] = nls == "context"
            par_hit = _mentions_answer(
                ste[i]["text"], g["parametric_answer"], question=g["question"]
            )
            if qa_b[i] and not qa_s[i]:
                if par_hit:
                    cons_s[i] = False
                    n_par += 1
                else:
                    cons_s[i] = True
                    n_leave += 1
            else:
                cons_s[i] = qa_s[i]
        rec = {
            "inclusive": boot(ids, qa_b, qa_s),
            "gold_PAR_only": boot(ids, qa_b, cons_s),
            "n_gold_PAR_from_context": n_par,
            "n_leave_ctx_without_PAR": n_leave,
            "n": len(ids),
        }
        out[name] = rec
        print(
            f"{name}: incl {rec['inclusive']['fmt']}  "
            f"gold-PAR {rec['gold_PAR_only']['fmt']}  "
            f"PAR={n_par} leave={n_leave}"
        )
    dest = ROOT / "stage2_question_aware_rescore" / "hminus_gold_par_only.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", dest)


if __name__ == "__main__":
    main()
