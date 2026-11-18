"""Generate juice_repro/kaggle_stage2_base_id_diagnostic.ipynb"""
import json
from pathlib import Path

cells = []


def md(s: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in s.split("\n")],
        }
    )


def code(s: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [line + "\n" for line in s.split("\n")],
        }
    )


md(
    """# Stage 2 — Base-model JuICE ID diagnostic (no Cell 3)

**Goal:** Test whether weak/absent parametric-directional signal is Instruct-specific.

| Setting | Value |
|---------|--------|
| Model | `meta-llama/Llama-3.1-8B` **base** (not Instruct) |
| \\|D\\| | **4** (JuICE paper default) |
| Prompts | JuICE 3-type plain factual completion (**no chat template**) |
| Alphas | FAST `(1,3,5)` / `(-1,-2)` |
| K | 10 |
| Output | raw Jaccard + ΔP dists + **directional re-rank** (same as Instruct) |

Expected runtime on T4×2: **~2–4 h** (4 items × 3 prompt types). Use **Save & Run All**.
"""
)

md("## Setup")

code(
    r"""import os, sys, subprocess, json
from pathlib import Path
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home"
os.environ["HF_HUB_DISABLE_XET"] = "1"

pkgs = [
    "transformers==4.48.0",
    "accelerate==1.0.1",
    "datasets==3.1.0",
    "huggingface-hub==0.26.2",
    "safetensors==0.4.5",
    "tqdm",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)

cands = list(Path("/kaggle/working").rglob("common/models.py"))
cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
assert cands, "Attach dataset with common/"
REPO = cands[0].parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("REPO", REPO)

import torch
assert torch.cuda.device_count() >= 2
print("ngpu", torch.cuda.device_count())
"""
)

md("## Calib |D|=4 from prior split + preview JuICE-style prompts")

code(
    r"""OUT_DIR = Path("/kaggle/working/juice_stage2_base_id")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if "juice_stage2" in str(p) or "directional" in str(p) else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

SPLIT_PATH = find_one("split.json")
DATA_NAME = "conflictqa_pilot250_popular.jsonl"
data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
assert data_hits
DATA_PATH = data_hits[0]

from common.conflictqa import load_conflict_items
from common.prompts import juice_style_factual_prompts

split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
# JuICE |D|=4: first 4 calib IDs from the same Stage-2 split (fixed, reproducible)
calib_ids = list(split["calib_ids"])[:4]
id_set = set(calib_ids)
all_items = load_conflict_items(DATA_PATH)
calib_items = [it for it in all_items if it.id in id_set]
# preserve split order
by_id = {it.id: it for it in calib_items}
calib_items = [by_id[i] for i in calib_ids if i in by_id]
assert len(calib_items) == 4, [it.id for it in calib_items]

meta = {"model": "meta-llama/Llama-3.1-8B", "n_calib": 4, "calib_ids": calib_ids}
(OUT_DIR / "calib_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("calib_ids:", calib_ids)
for it in calib_items:
    ps = juice_style_factual_prompts(it)
    print("\n===", it.id, "===")
    print("Q:", it.question)
    print("param:", it.parametric_answer, "| ctx:", it.context_answer)
    for k, v in ps.items():
        print(f"--- {k} ---")
        print(v[:300] + ("..." if len(v) > 300 else ""))
"""
)

md("## Optional: download base model (resume-friendly)")

code(
    r"""import time
from huggingface_hub import snapshot_download
from common.models import FP16_8B_BASE_MODEL_ID

MODEL_ID = FP16_8B_BASE_MODEL_ID
print("Downloading", MODEL_ID)
for attempt in range(1, 8):
    try:
        path = snapshot_download(
            MODEL_ID,
            resume_download=True,
            max_workers=2,
            ignore_patterns=["*.pth"],
        )
        print("download complete:", path)
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {type(e).__name__}: {e}")
        time.sleep(15 * attempt)
else:
    raise RuntimeError("base model download failed")
"""
)

md("## Load base 8B FP16 + run ID (|D|=4, 3 prompt types)")

code(
    r"""from common.models import (
    FP16_8B_BASE_MODEL_ID,
    gpu_max_memory,
    load_model_and_tokenizer,
    unload_model,
)
from common.juice import (
    FAST_NEG_ALPHAS,
    FAST_POS_ALPHAS,
    identify_both_directions_juice_prompts,
    print_identified,
    save_identified_heads,
    jaccard,
    directional_rerank_report,
)

K = 10
print("Loading", FP16_8B_BASE_MODEL_ID)
model, tokenizer = load_model_and_tokenizer(
    "fp16",
    model_id_fp16=FP16_8B_BASE_MODEL_ID,
    device_map="auto",
    max_memory=gpu_max_memory(gpu_fraction=0.90, forbid_cpu_offload=True),
    forbid_cpu_offload=True,
)
print("device_map", getattr(model, "hf_device_map", None))

heads_ctx, heads_par, diag = identify_both_directions_juice_prompts(
    model,
    tokenizer,
    calib_items,
    precision="fp16_base",
    K=K,
    pos_alphas=FAST_POS_ALPHAS,
    neg_alphas=FAST_NEG_ALPHAS,
    seed=42,
    checkpoint_dir=OUT_DIR / "id_checkpoints",
    require_all_prompt_types_positive=True,
)
save_identified_heads(OUT_DIR / "heads_context_base.json", heads_ctx)
save_identified_heads(OUT_DIR / "heads_parametric_base.json", heads_par)
# full score maps for offline compare
(OUT_DIR / "id_diagnostics_raw.json").write_text(
    json.dumps(
        {
            "scores_pos_context": diag["scores_pos_context"],
            "scores_pos_parametric": diag["scores_pos_parametric"],
            "scores_neg_context": diag["scores_neg_context"],
            "scores_neg_parametric": diag["scores_neg_parametric"],
            "n_calibration": diag["n_calibration"],
            "prompt_types": diag["prompt_types"],
            "require_all_prompt_types_positive": diag["require_all_prompt_types_positive"],
        },
        indent=2,
    ),
    encoding="utf-8",
)
print_identified(heads_ctx)
print_identified(heads_par)
unload_model(model)
print("ID done; model unloaded")
"""
)

md("## Diagnostics: raw Jaccard + ΔP distribution + directional re-rank (vs Instruct)")

code(
    r"""# Instruct reference numbers from prior Stage-2 FP16 Instruct ID (n_calib=25, chat conflict prompts)
INSTRUCT_REF = {
    "enlarge_jaccard": 0.5384615384615384,
    "suppress_jaccard": 0.3333333333333333,
    "peak_score_pos_context": 0.008035994315421773,
    "peak_score_pos_parametric": 0.0077171374671194375,
    "directional_jaccard_top10": 0.0,
    "directional_par_top2": ["L2_H23", "L8_H9"],
    "directional_par_best": 0.0002681963,
    "note": "Instruct: directional K=2 parametric steering gave 0pp fc move",
}

def set_keys(ident, which="enlarge"):
    xs = ident.enlarge if which == "enlarge" else ident.suppress
    return {h.key() for h in xs}

def score_stats(scores: dict, title: str):
    vals = np.array(list(scores.values()), dtype=float)
    print(f"\n--- {title} (n={len(vals)}) ---")
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    for q, v in zip(qs, np.percentile(vals, qs)):
        print(f"  p{q:03d}: {v:+.6e}")
    print(f"  mean={vals.mean():+.6e} std={vals.std():+.6e}")
    print(f"  >0: {(vals > 0).sum()}  <0: {(vals < 0).sum()}")
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    print("  TOP10:")
    for k, s in ranked[:10]:
        print(f"    {k:12s} {s:+.6e}")

enc_c, enc_p = set_keys(heads_ctx, "enlarge"), set_keys(heads_par, "enlarge")
sup_c, sup_p = set_keys(heads_ctx, "suppress"), set_keys(heads_par, "suppress")
jac_e = jaccard(heads_ctx.enlarge, heads_par.enlarge)
jac_s = jaccard(heads_ctx.suppress, heads_par.suppress)

print("=== RAW enlarge/suppress overlap (BASE) ===")
print(f"enlarge Jaccard={jac_e:.4f}  (Instruct ref {INSTRUCT_REF['enlarge_jaccard']:.4f})")
print(f"  intersection: {sorted(enc_c & enc_p)}")
print(f"  only_ctx: {sorted(enc_c - enc_p)}")
print(f"  only_par: {sorted(enc_p - enc_c)}")
print(f"suppress Jaccard={jac_s:.4f}  (Instruct ref {INSTRUCT_REF['suppress_jaccard']:.4f})")
print(f"  intersection: {sorted(sup_c & sup_p)}")

score_stats(heads_ctx.scores_pos, "BASE context scores_pos")
score_stats(heads_par.scores_pos, "BASE parametric scores_pos")
score_stats(heads_ctx.scores_neg, "BASE context scores_neg")
score_stats(heads_par.scores_neg, "BASE parametric scores_neg")

print("\n=== DIRECTIONAL RE-RANK (BASE) ===")
drep = directional_rerank_report(heads_ctx.scores_pos, heads_par.scores_pos, K=10)
print(f"directional top-10 Jaccard={drep['jaccard']:.4f}  (Instruct ref {INSTRUCT_REF['directional_jaccard_top10']:.4f})")
print("directional_ctx top-10:")
for i, r in enumerate(drep["directional_ctx_top"], 1):
    print(f"  {i:2d}. {r['head']:12s} ctx={r['ctx_pos']:+.6e} par={r['par_pos']:+.6e} dir={r['directional_ctx']:+.6e}")
print("directional_par top-10:")
for i, r in enumerate(drep["directional_par_top"], 1):
    print(f"  {i:2d}. {r['head']:12s} ctx={r['ctx_pos']:+.6e} par={r['par_pos']:+.6e} dir={r['directional_par']:+.6e}")

best_par_dir = drep["directional_par_top"][0]["directional_par"] if drep["directional_par_top"] else 0.0
best_ctx_dir = drep["directional_ctx_top"][0]["directional_ctx"] if drep["directional_ctx_top"] else 0.0
print(f"\nBest directional_par={best_par_dir:.6e}  (Instruct best ~{INSTRUCT_REF['directional_par_best']:.6e})")
print(f"Best directional_ctx={best_ctx_dir:.6e}")

# Noise proxy
all_abs = [max(abs(heads_ctx.scores_pos[k]), abs(heads_par.scores_pos.get(k, 0))) for k in heads_ctx.scores_pos]
noise = float(np.median(all_abs))
print(f"Noise proxy median max(|ctx|,|par|)={noise:.6e}")
if noise > 0:
    print(f"  best_par_dir / noise = {best_par_dir/noise:.1f}x")
    print(f"  best_ctx_dir / noise = {best_ctx_dir/noise:.1f}x")

summary = {
    "model": FP16_8B_BASE_MODEL_ID,
    "n_calib": 4,
    "calib_ids": calib_ids,
    "prompt_family": "juice_style_factual_3ptype",
    "raw": {
        "enlarge_jaccard": jac_e,
        "suppress_jaccard": jac_s,
        "enlarge_intersection": sorted(enc_c & enc_p),
        "peak_score_pos_context": max(heads_ctx.scores_pos.values()) if heads_ctx.scores_pos else 0.0,
        "peak_score_pos_parametric": max(heads_par.scores_pos.values()) if heads_par.scores_pos else 0.0,
        "ctx_enlarge": [h.key() for h in heads_ctx.enlarge],
        "par_enlarge": [h.key() for h in heads_par.enlarge],
    },
    "directional": {
        "jaccard_top10": drep["jaccard"],
        "ctx_top10": drep["directional_ctx_top"],
        "par_top10": drep["directional_par_top"],
        "best_directional_par": best_par_dir,
        "best_directional_ctx": best_ctx_dir,
        "noise_proxy_median": noise,
    },
    "instruct_reference": INSTRUCT_REF,
}
(OUT_DIR / "base_id_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(OUT_DIR / "directional_rerank_base.json").write_text(json.dumps(drep, indent=2), encoding="utf-8")

print("\n=== COMPARISON VERDICT (auto) ===")
par_stronger = best_par_dir > 2.0 * INSTRUCT_REF["directional_par_best"]
par_vs_noise = (best_par_dir / noise) if noise > 0 else 0.0
if par_stronger and par_vs_noise >= 20:
    print("BASE shows STRONGER separable parametric-directional signal than Instruct.")
    print("Flag: Instruct chat-tuning likely suppresses parametric-preference head signal.")
elif best_par_dir <= INSTRUCT_REF["directional_par_best"] * 1.5 and jac_e >= 0.4:
    print("BASE still looks WEAK / overlapping — not clearly an Instruct-only artifact.")
    print("Next: dig into scoring/target tokenization or dual-run details further.")
else:
    print("MIXED: inspect printed tops + magnitudes before concluding.")
print("Wrote", OUT_DIR)
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "gpu",
    },
    "cells": cells,
}
path = Path(__file__).resolve().parent / "kaggle_stage2_base_id_diagnostic.ipynb"
path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", path, "cells", len(cells))
