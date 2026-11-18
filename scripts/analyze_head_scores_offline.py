"""Offline analysis of Stage-2 JuICE head scores (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

P = Path(r"c:\Users\Tanjid\Downloads\results\juice_stage2_fp16")
if not (P / "heads_context.json").exists():
    P = Path(r"d:\tt") / ".."  # fallback unused
    # prefer local copy if downloads missing
for cand in [
    Path(r"c:\Users\Tanjid\Downloads\results\juice_stage2_fp16"),
    Path(r"d:\tt\juice_stage2_fp16"),
    Path(r"c:\Users\Tanjid\Downloads\results (1)3\juice_stage2_fp16"),
]:
    if (cand / "heads_context.json").exists():
        P = cand
        break

print("Using", P)
ctx = json.loads((P / "heads_context.json").read_text(encoding="utf-8"))
par = json.loads((P / "heads_parametric.json").read_text(encoding="utf-8"))


def key(h):
    return f"L{h['layer']}_H{h['head']}"


def set_of(heads):
    return {key(h) for h in heads}


def print_full(title: str, scores: dict, path: Path):
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    lines = [f"{k}\t{s:.12e}" for k, s in ranked]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{'='*70}\n{title}  n={len(ranked)}  FULL LIST written to {path}\n{'='*70}")
    print("(showing head/tail here; open the TSV for all 1024 rows)")
    for i, (k, s) in enumerate(ranked[:15], 1):
        print(f"{i:4d}. {k:12s}  {s:+.8e}")
    print("  ...")
    for i, (k, s) in enumerate(ranked[-10:], len(ranked) - 9):
        print(f"{i:4d}. {k:12s}  {s:+.8e}")


out = Path(r"d:\tt\juice_stage2_alpha_sweep") / "head_score_dumps"
out.mkdir(parents=True, exist_ok=True)

print_full("CONTEXT scores_pos (enlarge ranking)", ctx["scores_pos"], out / "context_scores_pos_all.tsv")
print_full("CONTEXT scores_neg (suppress ranking)", ctx["scores_neg"], out / "context_scores_neg_all.tsv")
print_full("PARAMETRIC scores_pos", par["scores_pos"], out / "parametric_scores_pos_all.tsv")
print_full("PARAMETRIC scores_neg", par["scores_neg"], out / "parametric_scores_neg_all.tsv")

enc_c, enc_p = set_of(ctx["enlarge"]), set_of(par["enlarge"])
sup_c, sup_p = set_of(ctx["suppress"]), set_of(par["suppress"])
shared_enc = enc_c & enc_p
only_par_enc = enc_p - enc_c
only_ctx_enc = enc_c - enc_p
shared_sup = sup_c & sup_p
only_par_sup = sup_p - sup_c
only_ctx_sup = sup_c - sup_p

print("\n" + "=" * 70)
print("2) PARAMETRIC-SPECIFIC H+ vs SHARED / OVERLAPPING")
print("=" * 70)
print(f"shared enlarge ({len(shared_enc)}): {sorted(shared_enc)}")
print(f"only parametric enlarge ({len(only_par_enc)}): {sorted(only_par_enc)}")
print(f"only context enlarge ({len(only_ctx_enc)}): {sorted(only_ctx_enc)}")


def rank_of(scores, k):
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    for i, (kk, s) in enumerate(ranked, 1):
        if kk == k:
            return i, s
    return None, None


print("\nPer-head: parametric scores_pos for selected enlarge heads")
print(f"{'head':12s} {'group':10s} {'par_pos':>12s} {'par_rank':>8s} {'ctx_pos':>12s} {'ctx_rank':>8s}")
for k in sorted(enc_p, key=lambda x: -par["scores_pos"].get(x, 0)):
    group = "SHARED" if k in shared_enc else "PAR_ONLY"
    pr, ps = rank_of(par["scores_pos"], k)
    cr, cs = rank_of(ctx["scores_pos"], k)
    print(f"{k:12s} {group:10s} {ps:+.6e} {pr:8d} {cs:+.6e} {cr:8d}")

print("\nCompare mean/max dP:")
for label, keys in [
    ("shared enlarge (on par_pos)", shared_enc),
    ("par-only enlarge (on par_pos)", only_par_enc),
    ("ctx-only enlarge (on ctx_pos)", only_ctx_enc),
    ("shared enlarge (on ctx_pos)", shared_enc),
]:
    sc = par["scores_pos"] if "par_pos" in label or label.startswith("shared enlarge (on par") else ctx["scores_pos"]
    if "ctx_pos" in label:
        sc = ctx["scores_pos"]
    vals = [sc[k] for k in keys]
    print(f"  {label}: n={len(vals)} mean={np.mean(vals):.6e} max={np.max(vals):.6e} min={np.min(vals):.6e}")

# How much of top-K parametric mass is shared vs unique?
par_ranked = sorted(par["scores_pos"].items(), key=lambda x: -x[1])
top10 = [k for k, _ in par_ranked[:10]]
print("\nParametric top-10 composition:")
for k, s in par_ranked[:10]:
    tag = "SHARED" if k in enc_c else "PAR_ONLY"
    in_sel = "IN_H+" if k in enc_p else "not_selected"
    print(f"  {k:12s} {s:+.6e}  {tag}  {in_sel}")

# If we took only non-overlapping heads ranked by par_pos among keys with par_pos>ctx_pos
print("\nHeads with par_pos > ctx_pos (directional preference), top 15:")
pref = []
for k in par["scores_pos"]:
    dp = par["scores_pos"][k]
    dc = ctx["scores_pos"].get(k, 0.0)
    pref.append((k, dp, dc, dp - dc))
pref.sort(key=lambda x: -x[3])
for k, dp, dc, diff in pref[:15]:
    print(f"  {k:12s} par={dp:+.6e} ctx={dc:+.6e} diff={diff:+.6e}")

print("\nHeads with ctx_pos > par_pos, top 15:")
pref2 = sorted(pref, key=lambda x: -(x[2] - x[1]))
for k, dp, dc, diff in pref2[:15]:
    print(f"  {k:12s} ctx={dc:+.6e} par={dp:+.6e} diff_ctx-par={dc-dp:+.6e}")

print("\n" + "=" * 70)
print("3) H- SUPPRESS SETS SIDE BY SIDE")
print("=" * 70)
print(f"{'head':12s} {'in_ctx_H-':9s} {'in_par_H-':9s} {'ctx_neg':>12s} {'par_neg':>12s} {'ctx_rank_neg':>12s} {'par_rank_neg':>12s}")
all_sup = sorted(sup_c | sup_p)
for k in all_sup:
    cr, cs = rank_of(ctx["scores_neg"], k)
    pr, ps = rank_of(par["scores_neg"], k)
    print(
        f"{k:12s} {str(k in sup_c):9s} {str(k in sup_p):9s} "
        f"{ctx['scores_neg'].get(k, float('nan')):+.6e} "
        f"{par['scores_neg'].get(k, float('nan')):+.6e} "
        f"{cr if cr else '-':>12} {pr if pr else '-':>12}"
    )

print("\nContext H- list + scores_neg:")
for h in ctx["suppress"]:
    k = key(h)
    rnk, s = rank_of(ctx["scores_neg"], k)
    print(f"  {k:12s}  ctx_neg={s:+.6e}  rank={rnk}")

print("\nParametric H- list + scores_neg:")
for h in par["suppress"]:
    k = key(h)
    rnk, s = rank_of(par["scores_neg"], k)
    print(f"  {k:12s}  par_neg={s:+.6e}  rank={rnk}")

# Distribution contrast H+ vs H−
def dist(name, scores):
    v = np.array(list(scores.values()), dtype=float)
    pos = v[v > 0]
    print(f"\n{name}:")
    print(f"  all  max={v.max():.6e} p99={np.percentile(v,99):.6e} p95={np.percentile(v,95):.6e} mean={v.mean():.6e}")
    if len(pos):
        print(f"  >0   n={len(pos)} max={pos.max():.6e} mean={pos.mean():.6e} median={np.median(pos):.6e}")
    # top-10 sum vs rest
    ranked = sorted(v, reverse=True)
    print(f"  sum top10={sum(ranked[:10]):.6e}  sum top3={sum(ranked[:3]):.6e}  sum all_pos={pos.sum() if len(pos) else 0:.6e}")


print("\n" + "=" * 70)
print("dP DISTRIBUTION: ENLARGE (pos) vs SUPPRESS (neg)")
print("=" * 70)
dist("context scores_pos", ctx["scores_pos"])
dist("context scores_neg", ctx["scores_neg"])
dist("parametric scores_pos", par["scores_pos"])
dist("parametric scores_neg", par["scores_neg"])

# Ratio of typical H+ vs H− selected scores
print("\nSelected-set score magnitudes:")
for name, keys, sc in [
    ("ctx H+ on scores_pos", enc_c, ctx["scores_pos"]),
    ("ctx H- on scores_neg", sup_c, ctx["scores_neg"]),
    ("par H+ on scores_pos", enc_p, par["scores_pos"]),
    ("par H- on scores_neg", sup_p, par["scores_neg"]),
]:
    vals = np.array([sc[k] for k in keys])
    print(f"  {name}: mean={vals.mean():.6e} median={np.median(vals):.6e} max={vals.max():.6e} min={vals.min():.6e}")

ratio_par = np.mean([par["scores_pos"][k] for k in enc_p]) / max(
    np.mean([par["scores_neg"][k] for k in sup_p]), 1e-12
)
ratio_ctx = np.mean([ctx["scores_pos"][k] for k in enc_c]) / max(
    np.mean([ctx["scores_neg"][k] for k in sup_c]), 1e-12
)
print(f"\nmean(H+ score) / mean(H- score): context={ratio_ctx:.1f}x  parametric={ratio_par:.1f}x")

print("\n" + "=" * 70)
print("4) VERDICT HINTS (auto)")
print("=" * 70)
par_only_vals = [par["scores_pos"][k] for k in only_par_enc]
shared_vals = [par["scores_pos"][k] for k in shared_enc]
print(f"par-only H+ max={max(par_only_vals):.6e}  shared H+ max={max(shared_vals):.6e}")
print(f"par-only / shared max ratio = {max(par_only_vals)/max(shared_vals):.3f}")
print(f"How many of parametric top-3 are shared? {[k for k,_ in par_ranked[:3]]}")
strong_par_unique = [k for k in only_par_enc if par["scores_pos"][k] >= 0.5 * max(shared_vals)]
print(f"par-only heads with score >= 50% of best shared: {strong_par_unique}")
