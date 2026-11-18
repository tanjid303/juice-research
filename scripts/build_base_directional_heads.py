"""Build directional head JSONs from base ID outputs for Cell-3 steering."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(r"d:\tt\juice_stage2_base_id")
OUT = Path(r"d:\tt\juice_stage2_base_id\directional_heads")
OUT.mkdir(parents=True, exist_ok=True)

raw = json.loads((BASE / "id_diagnostics_raw.json").read_text(encoding="utf-8"))
ctx_pos = raw["scores_pos_context"]
par_pos = raw["scores_pos_parametric"]

keys = sorted(set(ctx_pos) | set(par_pos))
rows = []
for k in keys:
    layer, head = k.replace("L", "").split("_H")
    rows.append(
        {
            "layer": int(layer),
            "head": int(head),
            "key": k,
            "ctx_pos": float(ctx_pos.get(k, 0.0)),
            "par_pos": float(par_pos.get(k, 0.0)),
            "directional_ctx": float(ctx_pos.get(k, 0.0)) - float(par_pos.get(k, 0.0)),
            "directional_par": float(par_pos.get(k, 0.0)) - float(ctx_pos.get(k, 0.0)),
        }
    )

by_ctx = sorted(rows, key=lambda r: -r["directional_ctx"])
by_par = sorted(rows, key=lambda r: -r["directional_par"])


def make_ident(selected: list[dict], steer_target: str, K: int, tag: str) -> dict:
    return {
        "enlarge": [{"layer": r["layer"], "head": r["head"]} for r in selected[:K]],
        "suppress": [],
        "scores_pos": ctx_pos if steer_target == "context" else par_pos,
        "scores_neg": {},
        "precision": "fp16_base",
        "K": K,
        "steer_target": steer_target,
        "n_calibration": 4,
        "pos_alphas": [],
        "neg_alphas": [],
        "selection_method": f"directional_rerank_base_{tag}",
    }


variants = [
    ("heads_dir_context_k10.json", by_ctx, "context", 10),
    ("heads_dir_context_k5.json", by_ctx, "context", 5),
    ("heads_dir_parametric_k2.json", by_par, "parametric", 2),
    ("heads_dir_parametric_k3.json", by_par, "parametric", 3),
]

# Cleaner parametric: positive par_pos and positive directional_par
clean_par = [r for r in by_par if r["par_pos"] > 0 and r["directional_par"] > 0]
clean_keys = ["L20_H14", "L26_H15"]
clean_selected: list[dict] = []
for ck in clean_keys:
    for r in by_par:
        if r["key"] == ck:
            clean_selected.append(r)
            break
for r in clean_par:
    if r not in clean_selected and len(clean_selected) < 2:
        clean_selected.append(r)

variants.append(("heads_dir_parametric_k2_clean.json", clean_selected, "parametric", 2))

for name, ranked, target, K in variants:
    payload = make_ident(ranked, target, K, name.replace(".json", ""))
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(name, "->", [f"L{h['layer']}_H{h['head']}" for h in payload["enlarge"]])

# Also copy split.json for packaging
split_src = Path(r"d:\tt\juice_stage2_alpha_sweep\directional_heads\split.json")
if split_src.exists():
    (OUT / "split.json").write_text(split_src.read_text(encoding="utf-8"), encoding="utf-8")

print("\nTop directional par (reference):")
for r in by_par[:5]:
    print(f"  {r['key']} par={r['par_pos']:+.4e} ctx={r['ctx_pos']:+.4e} dir={r['directional_par']:+.4e}")
print("\nClean par heads:")
for r in clean_selected:
    print(f"  {r['key']} par={r['par_pos']:+.4e} ctx={r['ctx_pos']:+.4e} dir={r['directional_par']:+.4e}")
