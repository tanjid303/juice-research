"""Generate juice_repro/kaggle_gemma2_base_id.ipynb — FP16 JuICE ID on Gemma-2-9B base."""
import json
from pathlib import Path

cells = []


def md(s: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in s.split("\n")]})


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
    """# Gemma-2-9B base — JuICE ID (FP16, no Cell 3)

Copy of the Llama Stage 2 ID protocol on **`google/gemma-2-9b` base**.

Does **not** need the INT4 checkpoint. That waits until Cell-3 / transfer.

| Setting | Value |
|---------|--------|
| Model | `google/gemma-2-9b` **base** |
| \\|D\\| | **4** (same `calib_ids[:4]` as Llama Stage 2) |
| Prompts | JuICE 3-type plain factual completion (no chat template) |
| Alphas | FAST `(1,3,5)` / `(-1,-2)` |
| K | 10 |
| Attn | eager (`GetHeadsGemma2Attention`) |

**~3–5 h** on GPU T4 ×2. Run All in the **draft**. Quick Save when it prints `SAVE VERSION NOW`.
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
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.51.3",
        "accelerate>=0.26.0,<2.0",
        "safetensors>=0.4.0",
        "huggingface-hub",
        "tqdm",
    ]
)

from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)

cands = list(Path("/kaggle/working").rglob("common/models.py"))
cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
assert cands, "Attach kaggle_juice_gemma2_base_id.zip"
REPO = cands[0].parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("REPO", REPO)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
assert torch.cuda.is_available()
if torch.cuda.device_count() < 2:
    print("WARNING: prefer T4 x2")
"""
)

md("## Calib |D|=4 (same split as Llama Stage 2)")
code(
    r"""OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_base_id")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ID = "google/gemma-2-9b"

def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

SPLIT_PATH = find_one("split.json")
DATA_NAME = "conflictqa_pilot250_popular.jsonl"
data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
assert data_hits, DATA_NAME
DATA_PATH = data_hits[0]

from common.conflictqa import load_conflict_items
from common.prompts import juice_style_factual_prompts

split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
calib_ids = list(split["calib_ids"])[:4]
id_set = set(calib_ids)
all_items = load_conflict_items(DATA_PATH)
by_id = {it.id: it for it in all_items if it.id in id_set}
calib_items = [by_id[i] for i in calib_ids if i in by_id]
assert len(calib_items) == 4, [it.id for it in calib_items]

meta = {"model": MODEL_ID, "n_calib": 4, "calib_ids": calib_ids, "split_seed": split.get("seed")}
(OUT_DIR / "calib_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("calib_ids:", calib_ids)
for it in calib_items:
    ps = juice_style_factual_prompts(it)
    print("\n===", it.id, "===")
    print("Q:", it.question)
    print("param:", it.parametric_answer, "| ctx:", it.context_answer)
    print("coherent head:", ps["coherent_conflict"][:180])
"""
)

md("## Load Gemma-2-9B FP16 + JuICE 3-type ID")
code(
    r"""from common.models import gpu_max_memory, load_model_and_tokenizer, unload_model
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
print("Loading", MODEL_ID)
model, tokenizer = load_model_and_tokenizer(
    "fp16",
    model_id_fp16=MODEL_ID,
    device_map="auto",
    max_memory=gpu_max_memory(gpu_fraction=0.90, forbid_cpu_offload=True),
    forbid_cpu_offload=True,
)
print("device_map", getattr(model, "hf_device_map", None))
print("attn", model.config._attn_implementation, "model_type", model.config.model_type)
assert model.config.model_type == "gemma2"

heads_ctx, heads_par, diag = identify_both_directions_juice_prompts(
    model,
    tokenizer,
    calib_items,
    precision="fp16_gemma2_base",
    K=K,
    pos_alphas=FAST_POS_ALPHAS,
    neg_alphas=FAST_NEG_ALPHAS,
    seed=42,
    checkpoint_dir=OUT_DIR / "id_checkpoints",
    require_all_prompt_types_positive=True,
)
save_identified_heads(OUT_DIR / "heads_context_base.json", heads_ctx)
save_identified_heads(OUT_DIR / "heads_parametric_base.json", heads_par)
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

md("## Directional re-rank + Cell-3 head files")
code(
    r"""def set_keys(ident, which="enlarge"):
    xs = ident.enlarge if which == "enlarge" else ident.suppress
    return {h.key() for h in xs}

enc_c, enc_p = set_keys(heads_ctx, "enlarge"), set_keys(heads_par, "enlarge")
jac_e = jaccard(heads_ctx.enlarge, heads_par.enlarge)
jac_s = jaccard(heads_ctx.suppress, heads_par.suppress)
print("enlarge Jaccard", jac_e, "intersect", sorted(enc_c & enc_p))
print("suppress Jaccard", jac_s)

drep = directional_rerank_report(heads_ctx.scores_pos, heads_par.scores_pos, K=10)
print("directional top-10 Jaccard", drep["jaccard"])
print("directional_ctx top-10:")
for i, r in enumerate(drep["directional_ctx_top"], 1):
    print(f"  {i:2d}. {r['head']:12s} ctx={r['ctx_pos']:+.6e} par={r['par_pos']:+.6e} dir={r['directional_ctx']:+.6e}")
print("directional_par top-10:")
for i, r in enumerate(drep["directional_par_top"], 1):
    print(f"  {i:2d}. {r['head']:12s} ctx={r['ctx_pos']:+.6e} par={r['par_pos']:+.6e} dir={r['directional_par']:+.6e}")

all_abs = [max(abs(heads_ctx.scores_pos[k]), abs(heads_par.scores_pos.get(k, 0))) for k in heads_ctx.scores_pos]
noise = float(np.median(all_abs))
print("noise proxy", noise)

heads_dir = OUT_DIR / "directional_heads"
heads_dir.mkdir(parents=True, exist_ok=True)

ctx_pos = diag["scores_pos_context"]
par_pos = diag["scores_pos_parametric"]
rows = []
for k in sorted(set(ctx_pos) | set(par_pos)):
    layer, head = k.replace("L", "").split("_H")
    rows.append({
        "layer": int(layer), "head": int(head), "key": k,
        "ctx_pos": float(ctx_pos.get(k, 0.0)),
        "par_pos": float(par_pos.get(k, 0.0)),
        "directional_ctx": float(ctx_pos.get(k, 0.0)) - float(par_pos.get(k, 0.0)),
        "directional_par": float(par_pos.get(k, 0.0)) - float(ctx_pos.get(k, 0.0)),
    })
by_ctx = sorted(rows, key=lambda r: -r["directional_ctx"])
by_par = sorted(rows, key=lambda r: -r["directional_par"])
clean_par = [r for r in by_par if r["par_pos"] > 0 and r["directional_par"] > 0][:2]

def dump_heads(path, selected, steer_target, K, tag):
    payload = {
        "enlarge": [{"layer": r["layer"], "head": r["head"]} for r in selected[:K]],
        "suppress": [],
        "scores_pos": {},
        "scores_neg": {},
        "precision": "fp16_gemma2_base",
        "K": K,
        "steer_target": steer_target,
        "n_calibration": 4,
        "pos_alphas": [1.0, 3.0, 5.0],
        "neg_alphas": [-1.0, -2.0],
        "selection_method": tag,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(path.name, "->", [r["key"] for r in selected[:K]])

dump_heads(heads_dir / "heads_dir_context_k10.json", by_ctx, "context", 10, "directional_ctx")
dump_heads(heads_dir / "heads_dir_context_k5.json", by_ctx, "context", 5, "directional_ctx")
dump_heads(heads_dir / "heads_dir_parametric_k2.json", by_par, "parametric", 2, "directional_par")
dump_heads(heads_dir / "heads_dir_parametric_k2_clean.json", clean_par, "parametric", 2, "directional_par_clean")
(heads_dir / "split.json").write_text(SPLIT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
(OUT_DIR / "directional_rerank_base.json").write_text(json.dumps(drep, indent=2), encoding="utf-8")

summary = {
    "model": MODEL_ID,
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
        "clean_par_k2": [r["key"] for r in clean_par],
        "noise_proxy_median": noise,
    },
}
(OUT_DIR / "base_id_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print()
print("=" * 60)
print("SAVE VERSION NOW — Quick Save, not Save & Run All")
print("Need: heads_dir_context_k10.json, heads_dir_parametric_k2_clean.json, id_diagnostics_raw.json")
print("folder:", OUT_DIR)
print("=" * 60)
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
out = Path(__file__).resolve().parent / "kaggle_gemma2_base_id.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("Wrote", out)
