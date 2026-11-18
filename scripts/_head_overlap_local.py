import json
from pathlib import Path
import numpy as np

p = Path(r"c:/Users/Tanjid/Downloads/results/juice_stage2_fp16")
c = json.loads((p / "heads_context.json").read_text(encoding="utf-8"))
r = json.loads((p / "heads_parametric.json").read_text(encoding="utf-8"))

def keys(heads):
    return {f"L{h['layer']}_H{h['head']}" for h in heads}

ec, ep = keys(c["enlarge"]), keys(r["enlarge"])
sc, sp = keys(c["suppress"]), keys(r["suppress"])
print("enlarge Jaccard", len(ec & ep) / len(ec | ep))
print("  intersection", sorted(ec & ep))
print("  only_ctx", sorted(ec - ep))
print("  only_par", sorted(ep - ec))
print("suppress Jaccard", len(sc & sp) / len(sc | sp))
print("  intersection", sorted(sc & sp))

for name, d in [
    ("ctx_pos", c["scores_pos"]),
    ("ctx_neg", c["scores_neg"]),
    ("par_pos", r["scores_pos"]),
    ("par_neg", r["scores_neg"]),
]:
    v = np.array(list(d.values()), dtype=float)
    print(
        name,
        f"max={v.max():.6e}",
        f"p95={np.percentile(v, 95):.6e}",
        f"mean={v.mean():.6e}",
        f"frac>0={float((v > 0).mean()):.3f}",
    )
