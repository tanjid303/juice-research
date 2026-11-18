"""Directional re-rank of existing JuICE head scores (offline, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

for cand in [
    Path(r"c:\Users\Tanjid\Downloads\results\juice_stage2_fp16"),
    Path(r"d:\tt\juice_stage2_fp16"),
]:
    if (cand / "heads_context.json").exists():
        P = cand
        break
else:
    raise SystemExit("heads_*.json not found")

ctx = json.loads((P / "heads_context.json").read_text(encoding="utf-8"))
par = json.loads((P / "heads_parametric.json").read_text(encoding="utf-8"))

# Union of all head keys (should be 1024)
keys = sorted(set(ctx["scores_pos"]) | set(par["scores_pos"]))
assert len(keys) == 1024, len(keys)

rows = []
for k in keys:
    cpos = float(ctx["scores_pos"].get(k, 0.0))
    ppos = float(par["scores_pos"].get(k, 0.0))
    rows.append(
        {
            "head": k,
            "ctx_pos": cpos,
            "par_pos": ppos,
            "directional_ctx": cpos - ppos,
            "directional_par": ppos - cpos,
        }
    )

by_dctx = sorted(rows, key=lambda r: -r["directional_ctx"])
by_dpar = sorted(rows, key=lambda r: -r["directional_par"])

UNIVERSAL = {"L31_H14", "L22_H6"}
KNOWN_PAR_PREF = {"L2_H23", "L8_H9", "L1_H23"}
KNOWN_CTX_ONLY = {"L11_H19", "L11_H24", "L11_H25"}


def rank_map(sorted_rows, score_key):
    return {r["head"]: (i + 1, r[score_key]) for i, r in enumerate(sorted_rows)}


r_dctx = rank_map(by_dctx, "directional_ctx")
r_dpar = rank_map(by_dpar, "directional_par")
r_cpos = rank_map(sorted(rows, key=lambda r: -r["ctx_pos"]), "ctx_pos")
r_ppos = rank_map(sorted(rows, key=lambda r: -r["par_pos"]), "par_pos")

out_dir = Path(r"d:\tt\juice_stage2_alpha_sweep\head_score_dumps")
out_dir.mkdir(parents=True, exist_ok=True)

# Full dumps
for name, sorted_rows, col in [
    ("directional_ctx_all.tsv", by_dctx, "directional_ctx"),
    ("directional_par_all.tsv", by_dpar, "directional_par"),
]:
    lines = [
        "rank\thead\tctx_pos\tpar_pos\tdirectional_score",
    ]
    for i, r in enumerate(sorted_rows, 1):
        lines.append(
            f"{i}\t{r['head']}\t{r['ctx_pos']:.12e}\t{r['par_pos']:.12e}\t{r[col]:.12e}"
        )
    (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_top(title, sorted_rows, dkey, n=10):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(
        f"{'rank':>4} {'head':12s} {'ctx_pos':>12s} {'par_pos':>12s} "
        f"{'dir_score':>12s}  notes"
    )
    for i, r in enumerate(sorted_rows[:n], 1):
        notes = []
        if r["head"] in UNIVERSAL:
            notes.append("UNIVERSAL_BOOSTER")
        if r["head"] in KNOWN_PAR_PREF:
            notes.append("was_par_only_H+")
        if r["head"] in KNOWN_CTX_ONLY:
            notes.append("was_ctx_only_H+")
        print(
            f"{i:4d} {r['head']:12s} {r['ctx_pos']:+.6e} {r['par_pos']:+.6e} "
            f"{r[dkey]:+.6e}  {','.join(notes)}"
        )


print_top("NEW top-10 by directional_ctx = ctx_pos - par_pos", by_dctx, "directional_ctx")
print_top("NEW top-10 by directional_par = par_pos - ctx_pos", by_dpar, "directional_par")

print("\n--- Where did the universal boosters go? ---")
for h in sorted(UNIVERSAL):
    print(
        f"  {h}: raw_ctx_rank={r_cpos[h][0]} raw_par_rank={r_ppos[h][0]} | "
        f"dir_ctx_rank={r_dctx[h][0]} (score={r_dctx[h][1]:+.6e}) | "
        f"dir_par_rank={r_dpar[h][0]} (score={r_dpar[h][1]:+.6e})"
    )

print("\n--- Known preference heads under directional ranking ---")
for h in sorted(KNOWN_PAR_PREF | KNOWN_CTX_ONLY):
    print(
        f"  {h}: dir_ctx_rank={r_dctx[h][0]} ({r_dctx[h][1]:+.6e}) | "
        f"dir_par_rank={r_dpar[h][0]} ({r_dpar[h][1]:+.6e}) | "
        f"ctx_pos={next(r['ctx_pos'] for r in rows if r['head']==h):+.6e} "
        f"par_pos={next(r['par_pos'] for r in rows if r['head']==h):+.6e}"
    )

# Magnitude / noise check
dctx_scores = np.array([r["directional_ctx"] for r in rows])
dpar_scores = np.array([r["directional_par"] for r in rows])
# note directional_par = -directional_ctx for every head, so distributions are mirrors


def mag_report(name, top_rows, dkey, all_scores):
    top = np.array([r[dkey] for r in top_rows[:10]])
    print(f"\n{name}:")
    print(f"  top10 dir: max={top.max():.6e} min={top.min():.6e} mean={top.mean():.6e}")
    print(
        f"  all-head dir: mean={all_scores.mean():.6e} std={all_scores.std():.6e} "
        f"p95={np.percentile(all_scores, 95):.6e} p99={np.percentile(all_scores, 99):.6e}"
    )
    # how many std above mean?
    z = (top - all_scores.mean()) / (all_scores.std() + 1e-15)
    print(f"  top10 z-scores vs all-head dist: mean_z={z.mean():.2f} min_z={z.min():.2f} max_z={z.max():.2f}")
    # compare to original raw top10 pos magnitudes
    print(f"  top10 |dir| / original peak raw pos (~8e-3): {[f'{abs(x)/8.036e-3:.3f}' for x in top]}")
    # fraction of original top3 raw mass
    print(f"  sum(top10 dir)={top.sum():.6e}  vs sum(raw top3 ctx_pos)~1.70e-2")


mag_report("directional_ctx top-10 magnitude", by_dctx, "directional_ctx", dctx_scores)
mag_report("directional_par top-10 magnitude", by_dpar, "directional_par", dpar_scores)

# Because dpar = -dctx, top-10 par and top-10 ctx are disjoint by construction
# (opposite ends of same ranking). Jaccard of top-10 sets should be 0 unless ties.
set_ctx = {r["head"] for r in by_dctx[:10]}
set_par = {r["head"] for r in by_dpar[:10]}
inter = set_ctx & set_par
union = set_ctx | set_par
jac = len(inter) / len(union) if union else 1.0
print("\n" + "=" * 78)
print("JACCARD: directional top-10 ctx vs directional top-10 par")
print("=" * 78)
print(f"  |ctx_top10|={len(set_ctx)} |par_top10|={len(set_par)}")
print(f"  intersection ({len(inter)}): {sorted(inter)}")
print(f"  Jaccard={jac:.4f}  (original enlarge Jaccard was 0.538)")
print(f"  only_ctx: {sorted(set_ctx - set_par)}")
print(f"  only_par: {sorted(set_par - set_ctx)}")

# Sanity: directional_par == -directional_ctx
assert all(abs(r["directional_par"] + r["directional_ctx"]) < 1e-15 for r in rows)

# Is signal "real"? Compare top directional to typical absolute noise scale of raw scores
# Use median |raw| among mid-ranked heads as noise proxy
abs_raw = np.array([max(abs(r["ctx_pos"]), abs(r["par_pos"])) for r in rows])
noise = np.median(abs_raw)
print("\n--- Separability vs noise ---")
print(f"  median max(|ctx_pos|,|par_pos|) across heads (noise proxy): {noise:.6e}")
print(f"  best directional_ctx: {by_dctx[0]['directional_ctx']:.6e}  ({by_dctx[0]['directional_ctx']/noise:.1f}x noise proxy)")
print(f"  best directional_par: {by_dpar[0]['directional_par']:.6e}  ({by_dpar[0]['directional_par']/noise:.1f}x noise proxy)")
print(f"  10th directional_ctx: {by_dctx[9]['directional_ctx']:.6e}  ({by_dctx[9]['directional_ctx']/noise:.1f}x)")
print(f"  10th directional_par: {by_dpar[9]['directional_par']:.6e}  ({by_dpar[9]['directional_par']/noise:.1f}x)")

# Original raw top enlarge for reference
print("\n--- ORIGINAL raw-top enlarge (for comparison) ---")
print("ctx enlarge:", [f"L{h['layer']}_H{h['head']}" for h in ctx["enlarge"]])
print("par enlarge:", [f"L{h['layer']}_H{h['head']}" for h in par["enlarge"]])

# Save new top-10 JSON for potential Cell3 reuse
payload = {
    "method": "directional_rerank_offline",
    "directional_ctx_top10": [
        {
            "head": r["head"],
            "ctx_pos": r["ctx_pos"],
            "par_pos": r["par_pos"],
            "directional_ctx": r["directional_ctx"],
        }
        for r in by_dctx[:10]
    ],
    "directional_par_top10": [
        {
            "head": r["head"],
            "ctx_pos": r["ctx_pos"],
            "par_pos": r["par_pos"],
            "directional_par": r["directional_par"],
        }
        for r in by_dpar[:10]
    ],
    "jaccard_directional_top10": jac,
    "original_enlarge_jaccard": 0.5384615384615384,
    "universal_booster_ranks": {
        h: {
            "dir_ctx_rank": r_dctx[h][0],
            "dir_par_rank": r_dpar[h][0],
            "directional_ctx": r_dctx[h][1],
            "directional_par": r_dpar[h][1],
        }
        for h in UNIVERSAL
    },
}
(out_dir / "directional_rerank_top10.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("\nWrote", out_dir / "directional_rerank_top10.json")
print("Full rankings:", out_dir / "directional_ctx_all.tsv", out_dir / "directional_par_all.tsv")
