"""Build IdentifiedHeads JSONs from directional re-rank (offline)."""
from __future__ import annotations

import json
from pathlib import Path

src = Path(r"d:\tt\juice_stage2_alpha_sweep\head_score_dumps\directional_rerank_top10.json")
# Prefer fresh recompute from full scores if dump missing pieces
heads_dir = Path(r"c:\Users\Tanjid\Downloads\results\juice_stage2_fp16")
ctx = json.loads((heads_dir / "heads_context.json").read_text(encoding="utf-8"))
par = json.loads((heads_dir / "heads_parametric.json").read_text(encoding="utf-8"))

keys = sorted(set(ctx["scores_pos"]) | set(par["scores_pos"]))
rows = []
for k in keys:
    cpos = float(ctx["scores_pos"][k])
    ppos = float(par["scores_pos"][k])
    layer, head = k.replace("L", "").split("_H")
    rows.append(
        {
            "layer": int(layer),
            "head": int(head),
            "key": k,
            "ctx_pos": cpos,
            "par_pos": ppos,
            "directional_ctx": cpos - ppos,
            "directional_par": ppos - cpos,
        }
    )

by_ctx = sorted(rows, key=lambda r: -r["directional_ctx"])
by_par = sorted(rows, key=lambda r: -r["directional_par"])


def make_ident(selected, steer_target: str, score_field: str, K: int) -> dict:
    scores = {r["key"]: r[score_field] for r in rows}
    # sort scores descending for JSON readability
    scores_sorted = dict(sorted(scores.items(), key=lambda x: -x[1]))
    return {
        "enlarge": [{"layer": r["layer"], "head": r["head"]} for r in selected[:K]],
        "suppress": [],
        "scores_pos": scores_sorted,
        "scores_neg": {},
        "precision": "fp16",
        "K": K,
        "steer_target": steer_target,
        "n_calibration": 25,
        "pos_alphas": [],
        "neg_alphas": [],
        "selection_method": "directional_rerank_offline",
        "score_field": score_field,
    }


out = Path(r"d:\tt\juice_stage2_alpha_sweep\directional_heads")
out.mkdir(parents=True, exist_ok=True)

variants = [
    ("heads_dir_context_k10.json", by_ctx, "context", "directional_ctx", 10),
    ("heads_dir_context_k5.json", by_ctx, "context", "directional_ctx", 5),
    ("heads_dir_parametric_k2.json", by_par, "parametric", "directional_par", 2),
    ("heads_dir_parametric_k3.json", by_par, "parametric", "directional_par", 3),
]

for name, ranked, target, field, K in variants:
    payload = make_ident(ranked, target, field, K)
    path = out / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    keys_sel = [f"L{h['layer']}_H{h['head']}" for h in payload["enlarge"]]
    print(f"{name}: {keys_sel}")

# copy split for packaging convenience
split_src = heads_dir / "split.json"
if split_src.exists():
    (out / "split.json").write_text(split_src.read_text(encoding="utf-8"), encoding="utf-8")
    print("copied split.json")
