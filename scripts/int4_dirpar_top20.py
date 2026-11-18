"""INT4 directional_par ranks 1-20 + positive par_pos scan (local, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(r"d:\tt")
raw = json.loads((ROOT / "juice_stage2_int4_native_id" / "id_diagnostics_raw.json").read_text(encoding="utf-8"))
ctx, par = raw["scores_pos_context"], raw["scores_pos_parametric"]
keys = set(ctx) | set(par)
rows = []
for k in keys:
    c = float(ctx.get(k, 0.0))
    p = float(par.get(k, 0.0))
    rows.append({"head": k, "ctx_pos": c, "par_pos": p, "dir_par": p - c})
rows.sort(key=lambda r: -r["dir_par"])

absmax = [max(abs(ctx.get(k, 0.0)), abs(par.get(k, 0.0))) for k in keys]
noise = float(np.median(absmax))
thr5 = 5 * noise

print("noise proxy median max(|ctx|,|par|)=%.4e  5x=%.4e" % (noise, thr5))
print()
print("=== directional_par ranks 1-20 ===")
print("%3s %-10s %10s %12s %12s  %s" % ("rk", "head", "dir_par", "par_pos", "ctx_pos", "flags"))
top20 = []
for i, r in enumerate(rows[:20], 1):
    flags = []
    if r["par_pos"] > 0:
        flags.append("par_pos>0")
    if r["par_pos"] > 0 and r["dir_par"] > 0:
        flags.append("CLEAN")
    if r["par_pos"] >= thr5:
        flags.append("par>=5x_noise")
    line = "%3d %-10s %+10.4f %+12.4e %+12.4e  %s" % (
        i, r["head"], r["dir_par"], r["par_pos"], r["ctx_pos"], " ".join(flags)
    )
    print(line)
    top20.append({**r, "rank": i, "flags": flags})

pos = [r for r in rows if r["par_pos"] > 0]
print()
print("=== par_pos>0 heads ranked by directional_par (top 15 of %d) ===" % len(pos))
for i, r in enumerate(pos[:15], 1):
    mag = "MEANINGFUL" if r["par_pos"] >= thr5 else "near-noise"
    rk = next(j for j, x in enumerate(rows, 1) if x["head"] == r["head"])
    print(
        "%3d %-10s dir_rank=%3d dir=%+.4f par=%+.4e ctx=%+.4e %s"
        % (i, r["head"], rk, r["dir_par"], r["par_pos"], r["ctx_pos"], mag)
    )

print()
print("=== named heads ===")
for k in ["L20_H14", "L26_H15", "L31_H14", "L16_H30", "L17_H27", "L8_H9"]:
    for i, r in enumerate(rows, 1):
        if r["head"] == k:
            print(
                "  %-10s rank=%4d dir=%+.4f par=%+.4e ctx=%+.4e"
                % (k, i, r["dir_par"], r["par_pos"], r["ctx_pos"])
            )
            break

out = {
    "noise_proxy_median": noise,
    "par_pos_meaningful_threshold_5x": thr5,
    "directional_par_top20": top20,
    "par_pos_positive_top15_by_dir_par": [
        {**r, "dir_rank": next(j for j, x in enumerate(rows, 1) if x["head"] == r["head"])}
        for r in pos[:15]
    ],
    "n_par_pos": len(pos),
    "n_heads": len(rows),
}
path = ROOT / "juice_stage2_int4_native_id" / "directional_par_top20.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nwrote", path)
