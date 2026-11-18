import json
from pathlib import Path
import numpy as np

p = Path(r"d:\tt\juice_stage2_base_id")
s = json.loads((p / "base_id_comparison_summary.json").read_text(encoding="utf-8"))
ctx = json.loads((p / "heads_context_base.json").read_text(encoding="utf-8"))
par = json.loads((p / "heads_parametric_base.json").read_text(encoding="utf-8"))

print("calib", s["calib_ids"])
print("RAW enlarge jac", s["raw"]["enlarge_jaccard"])
print("RAW suppress jac", s["raw"]["suppress_jaccard"])
print("peak ctx", s["raw"]["peak_score_pos_context"], "peak par", s["raw"]["peak_score_pos_parametric"])
print("ctx enlarge", s["raw"]["ctx_enlarge"])
print("par enlarge", s["raw"]["par_enlarge"])
print("DIR jac", s["directional"]["jaccard_top10"])
print("best dir par", s["directional"]["best_directional_par"], "best dir ctx", s["directional"]["best_directional_ctx"])
print("noise", s["directional"]["noise_proxy_median"])
bp = s["directional"]["best_directional_par"]
bc = s["directional"]["best_directional_ctx"]
n = s["directional"]["noise_proxy_median"]
print("par/noise", bp / n if n else None, "ctx/noise", bc / n if n else None)

print("\nDIR par top5:")
for r in s["directional"]["par_top10"][:5]:
    print(r)
print("\nDIR ctx top5:")
for r in s["directional"]["ctx_top10"][:5]:
    print(r)

# Compare to instruct
ref = s["instruct_reference"]
print("\nvs Instruct enlarge jac", ref["enlarge_jaccard"], "-> base", s["raw"]["enlarge_jaccard"])
print("vs Instruct peak par", ref["peak_score_pos_parametric"], "-> base", s["raw"]["peak_score_pos_parametric"])
print("vs Instruct best dir par", ref["directional_par_best"], "-> base", s["directional"]["best_directional_par"])
print("ratio best_dir_par base/instruct", s["directional"]["best_directional_par"] / ref["directional_par_best"])

# How many enlarge heads overlap between ctx and par
ec, ep = set(s["raw"]["ctx_enlarge"]), set(s["raw"]["par_enlarge"])
print("enlarge inter", sorted(ec & ep), "only_par", sorted(ep - ec), "only_ctx", sorted(ec - ep))
