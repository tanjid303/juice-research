"""Local analysis of native INT4 ID vs FP16 heads + Jaccard noise floor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"d:\tt")
sys.path.insert(0, str(ROOT))

from common.juice import (  # noqa: E402
    bootstrap_id_jaccard,
    directional_par_clean_keys,
    directional_rerank_report,
    directional_top_keys,
    jaccard,
    load_identified_heads,
    rank_of_key,
)

INT4 = ROOT / "juice_stage2_int4_native_id"
FP16_HEADS = ROOT / "juice_stage2_base_id" / "directional_heads"
OUT = INT4 / "int4_native_id_local_analysis.json"


def keys_of(ident) -> list[str]:
    return [h.key() for h in ident.enlarge]


def main() -> None:
    ctx = load_identified_heads(INT4 / "heads_context_int4.json")
    par = load_identified_heads(INT4 / "heads_parametric_int4.json")
    raw = json.loads((INT4 / "id_diagnostics_raw.json").read_text(encoding="utf-8"))
    per_item = json.loads((INT4 / "id_per_item_scores.json").read_text(encoding="utf-8"))

    ctx_pos = raw["scores_pos_context"]
    par_pos = raw["scores_pos_parametric"]
    drep = directional_rerank_report(ctx_pos, par_pos, K=10)
    (INT4 / "directional_rerank_int4.json").write_text(json.dumps(drep, indent=2), encoding="utf-8")

    ctx10 = [r["head"] for r in drep["directional_ctx_top"]]
    par_dir = [r["head"] for r in drep["directional_par_top"]]
    par2_clean = directional_par_clean_keys(ctx_pos, par_pos, K=2)

    fp16 = {
        "dir_context_k10": keys_of(load_identified_heads(FP16_HEADS / "heads_dir_context_k10.json")),
        "dir_context_k5": keys_of(load_identified_heads(FP16_HEADS / "heads_dir_context_k5.json")),
        "dir_parametric_k2": keys_of(load_identified_heads(FP16_HEADS / "heads_dir_parametric_k2.json")),
        "dir_parametric_k3": keys_of(load_identified_heads(FP16_HEADS / "heads_dir_parametric_k3.json")),
        "dir_parametric_k2_clean": keys_of(
            load_identified_heads(FP16_HEADS / "heads_dir_parametric_k2_clean.json")
        ),
    }
    int4_sets = {
        "dir_context_k10": ctx10,
        "dir_context_k5": ctx10[:5],
        "dir_parametric_k2": par_dir[:2],
        "dir_parametric_k3": par_dir[:3],
        "dir_parametric_k2_clean": par2_clean,
    }

    fp16_raw_ctx = json.loads(
        (ROOT / "juice_stage2_base_id" / "heads_context_base.json").read_text(encoding="utf-8")
    )
    fp16_raw_par = json.loads(
        (ROOT / "juice_stage2_base_id" / "heads_parametric_base.json").read_text(encoding="utf-8")
    )
    fp16_raw_ctx_keys = [f"L{h['layer']}_H{h['head']}" for h in fp16_raw_ctx["enlarge"]]
    fp16_raw_par_keys = [f"L{h['layer']}_H{h['head']}" for h in fp16_raw_par["enlarge"]]
    int4_raw_ctx = keys_of(ctx)
    int4_raw_par = keys_of(par)

    def jac(a, b):
        sa, sb = set(a), set(b)
        return (len(sa & sb) / len(sa | sb)) if (sa or sb) else 1.0

    canonical = ["L20_H14", "L26_H15", "L31_H14"]
    ranks = {}
    for k in canonical:
        ranks[k] = {
            "rank_directional_par": rank_of_key(ctx_pos, par_pos, k, "parametric"),
            "rank_directional_ctx": rank_of_key(ctx_pos, par_pos, k, "context"),
            "ctx_pos": ctx_pos.get(k),
            "par_pos": par_pos.get(k),
            "dir_par": float(par_pos.get(k, 0.0)) - float(ctx_pos.get(k, 0.0)),
            "in_int4_raw_par_enlarge": k in set(int4_raw_par),
            "in_int4_raw_ctx_enlarge": k in set(int4_raw_ctx),
        }

    fp16_ctx_ranks = {}
    for k in fp16["dir_context_k10"]:
        fp16_ctx_ranks[k] = {
            "rank_directional_ctx": rank_of_key(ctx_pos, par_pos, k, "context"),
            "ctx_pos": ctx_pos.get(k),
            "par_pos": par_pos.get(k),
        }

    jac_boot = bootstrap_id_jaccard(
        per_item,
        fp16_sets=fp16,
        int4_sets=int4_sets,
        n_boot=25,
        seed=42,
    )
    # compact: drop long value lists for print
    (INT4 / "jaccard_bootstrap.json").write_text(json.dumps(jac_boot, indent=2), encoding="utf-8")

    summary = {
        "n_calib_items_per_item_file": len(per_item),
        "item_ids": [r["id"] for r in per_item],
        "raw": {
            "int4_ctx_enlarge": int4_raw_ctx,
            "int4_par_enlarge": int4_raw_par,
            "fp16_ctx_enlarge": fp16_raw_ctx_keys,
            "fp16_par_enlarge": fp16_raw_par_keys,
            "jaccard_ctx_fp16_vs_int4": jac(fp16_raw_ctx_keys, int4_raw_ctx),
            "jaccard_par_fp16_vs_int4": jac(fp16_raw_par_keys, int4_raw_par),
            "overlap_ctx": sorted(set(fp16_raw_ctx_keys) & set(int4_raw_ctx)),
            "overlap_par": sorted(set(fp16_raw_par_keys) & set(int4_raw_par)),
            "int4_ctx_vs_par_jaccard": jaccard(ctx.enlarge, par.enlarge),
            "peak_ctx": max(ctx_pos.values()),
            "peak_par": max(par_pos.values()),
        },
        "directional": int4_sets,
        "fp16_sets": fp16,
        "canonical_parametric": ranks,
        "fp16_context_heads_on_int4": fp16_ctx_ranks,
        "directional_int4_vs_fp16_jaccard": {k: jac(fp16[k], int4_sets[k]) for k in fp16},
        "jaccard_bootstrap": {
            spec: {
                "observed_fp16_vs_int4": row["observed_fp16_vs_int4"],
                "noise_int4_vs_int4_mean": row["bootstrap_int4_vs_int4"]["mean"],
                "noise_int4_vs_int4_ci95": row["bootstrap_int4_vs_int4"]["ci95"],
                "bootstrap_vs_int4_id_mean": row["bootstrap_vs_int4_id"]["mean"],
            }
            for spec, row in jac_boot["by_spec"].items()
        },
        "random_k": "NOT IN DOWNLOAD — cell 7 did not land in juice_stage2_int4_native_id/",
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== RAW enlarge ===")
    print("INT4 ctx", int4_raw_ctx)
    print("FP16 ctx", fp16_raw_ctx_keys)
    print("  overlap", sorted(set(fp16_raw_ctx_keys) & set(int4_raw_ctx)), "J=", jac(fp16_raw_ctx_keys, int4_raw_ctx))
    print("INT4 par", int4_raw_par)
    print("FP16 par", fp16_raw_par_keys)
    print("  overlap", sorted(set(fp16_raw_par_keys) & set(int4_raw_par)), "J=", jac(fp16_raw_par_keys, int4_raw_par))

    print("\n=== DIRECTIONAL ===")
    print("INT4 ctx K10", ctx10)
    print("FP16 ctx K10", fp16["dir_context_k10"])
    print("  overlap", sorted(set(ctx10) & set(fp16["dir_context_k10"])), "J=", jac(ctx10, fp16["dir_context_k10"]))
    print("INT4 ctx K5 ", ctx10[:5])
    print("FP16 ctx K5 ", fp16["dir_context_k5"])
    print("  overlap", sorted(set(ctx10[:5]) & set(fp16["dir_context_k5"])), "J=", jac(ctx10[:5], fp16["dir_context_k5"]))
    print("INT4 par top10", par_dir)
    print("INT4 par K2 ", par_dir[:2], "K3", par_dir[:3])
    print("INT4 par clean K2", par2_clean)
    print("FP16 clean K2", fp16["dir_parametric_k2_clean"])

    print("\n=== Canonical parametric ranks on INT4 ===")
    for k, r in ranks.items():
        print(f"  {k}: dir_par_rank={r['rank_directional_par']} ctx_pos={r['ctx_pos']} par_pos={r['par_pos']} dir_par={r['dir_par']:.4f}")

    print("\n=== FP16 context heads: INT4 directional_ctx rank ===")
    for k, r in fp16_ctx_ranks.items():
        print(f"  {k:10s} rank={r['rank_directional_ctx']} ctx_pos={r['ctx_pos']}")

    print("\n=== Jaccard vs |D|=4 noise floor (25 draws) ===")
    for spec, row in jac_boot["by_spec"].items():
        noise = row["bootstrap_int4_vs_int4"]
        obs = row["observed_fp16_vs_int4"]
        lo, hi = noise["ci95"]
        inside = lo <= obs <= hi
        print(
            f"  {spec:28s} obs J={obs:.3f}  INT4-vs-INT4 noise mean={noise['mean']:.3f} "
            f"[{lo:.3f}, {hi:.3f}]  {'WITHIN noise' if inside else 'OUTSIDE noise'}"
        )

    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
