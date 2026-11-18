from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

N, SEED = 100, 42
ROOT = Path("/kaggle/working")
DIRS = {
 "fp16": ROOT / "gemma_n100_fp16",
 "int4": ROOT / "gemma_n100_int4_headline",
 "fp16_id_hminus": ROOT / "gemma_n100_fp16_hminus",
 "fp16_rand_hminus": ROOT / "gemma_n100_fp16_random_hminus",
 "int4_hminus": ROOT / "gemma_n100_int4_hminus",
}

def rows(p):
    return {r["id"]: r for r in json.loads(p.read_text(encoding="utf-8"))}
def fc(r): return r["label"] == "context"
def rate(ids, tab): return sum(fc(tab[i]) for i in ids) / len(ids)
def pco(tab):
    return {k: sum(r["label"] == k for r in tab.values())
            for k in ("parametric", "context", "other")}
def coherent(text):
    t = (text or "").strip()
    if len(t) < 2 or t.count("://") >= 3:
        return False
    import re
    if re.fullmatch(r"[\W\d_]+", t):
        return False
    toks = t.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters >= max(2, int(0.3 * len(t)))
def coh(tab):
    return sum(coherent(r.get("text")) for r in tab.values()) / len(tab)
def ci(xs):
    lo, hi = np.percentile(xs, [2.5, 97.5])
    return [100*float(lo), 100*float(hi)]
def stat(point, draws):
    lo, hi = ci(draws)
    return {"delta_pp": 100*point, "ci95_pp": [lo, hi],
            "excludes_zero": not (lo <= 0 <= hi)}

fp = DIRS["fp16"]; i4 = DIRS["int4"]
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
 "fp16_hminus_id": rows(DIRS["fp16_id_hminus"] / "suppress_context_k10.json"),
 "fp16_hminus_rand": rows(DIRS["fp16_rand_hminus"] / "random_suppress_k10.json"),
 "int4_hminus_id": rows(DIRS["int4_hminus"] / "suppress_context_k10.json"),
 "int4_hminus_rand": rows(DIRS["int4_hminus"] / "random_suppress_k10.json"),
}
ids = sorted(set.intersection(*(set(x) for x in tabs.values())))
assert len(ids) == N, len(ids)
rng = np.random.default_rng(SEED)
draws = [[ids[j] for j in rng.integers(0, N, size=N)] for _ in range(1000)]
def delta(a, b, d): return rate(d, tabs[a]) - rate(d, tabs[b])
def boot(a, b, base, name):
    point = delta(a, base, ids)
    xs = [delta(a, base, d) for d in draws]
    return {"condition": name, "fc": rate(ids, tabs[a]), "baseline_fc": rate(ids, tabs[base]),
            "pco": pco(tabs[a]), "coherence": coh(tabs[a]), **stat(point, xs)}
out = []
out += [boot("fp16_context_k10","fp16_baseline","fp16_baseline","FP16 context K10")]
out += [boot("int4_context_k10","int4_baseline","int4_baseline","INT4 context K10 transfer")]
out += [boot("int4_random_k10","int4_baseline","int4_baseline","INT4 random K10")]
out += [boot("fp16_parametric_k2_clean","fp16_baseline","fp16_baseline","FP16 clean parametric K2")]
out += [boot("fp16_raw_l27_h6","fp16_baseline","fp16_baseline","FP16 raw L27_H6")]
out += [boot("int4_raw_l27_h6","int4_baseline","int4_baseline","INT4 raw L27_H6 transfer")]
out += [boot("int4_random_k1","int4_baseline","int4_baseline","INT4 random K1")]
out += [boot("fp16_hminus_id","fp16_baseline","fp16_baseline","FP16 identified H-")]
out += [boot("fp16_hminus_rand","fp16_baseline","fp16_baseline","FP16 random H-")]
out += [boot("int4_hminus_id","int4_baseline","int4_baseline","INT4 identified H-")]
out += [boot("int4_hminus_rand","int4_baseline","int4_baseline","INT4 random H-")]
def spec(idk, randk, basek):
    point = delta(idk, basek, ids)-delta(randk, basek, ids)
    xs = [delta(idk,basek,d)-delta(randk,basek,d) for d in draws]
    return stat(point,xs)
spec16 = spec("fp16_hminus_id","fp16_hminus_rand","fp16_baseline")
spec4 = spec("int4_hminus_id","int4_hminus_rand","int4_baseline")
did_point = (delta("int4_hminus_id","int4_baseline",ids)-delta("int4_hminus_rand","int4_baseline",ids)
             -delta("fp16_hminus_id","fp16_baseline",ids)+delta("fp16_hminus_rand","fp16_baseline",ids))
did_draws = [delta("int4_hminus_id","int4_baseline",d)-delta("int4_hminus_rand","int4_baseline",d)
             -delta("fp16_hminus_id","fp16_baseline",d)+delta("fp16_hminus_rand","fp16_baseline",d) for d in draws]
payload = {"n":N,"n_boot":1000,"seed":SEED,"ids":ids,"cells":out,
           "specificity_fp16":spec16,"specificity_int4":spec4,
           "difference_in_differences":stat(did_point,did_draws)}
(ROOT/"gemma_n100_raw_results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
with (ROOT/"gemma_n100_raw_results.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["condition","fc","delta_pp","ci95_pp","excludes_zero","pco","coherence"])
    w.writeheader()
    for x in out: w.writerow(x)
print(json.dumps(payload,indent=2))
print("Wrote gemma_n100_raw_results.json and .csv")
