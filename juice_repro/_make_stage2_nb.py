"""Generate juice_repro/kaggle_stage2_fp16_8b.ipynb"""
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
    """# Stage 2 — JuICE FP16 reproduction on Llama-3.1-8B-Instruct (Week 2 gate)

**Real dual-run** (ported from GaotangLi/JUICE), not the old prefill-only approximation.

## Before Run All
1. Accelerator: **GPU T4 x2**
2. Internet on; accept Llama-3.1-8B-Instruct license; Kaggle secret `HF_TOKEN`
3. Attach this repo as a Dataset (must include `common/` + `data/conflictqa_pilot250_popular.jsonl`)

## Method note
- Port of JuICE: post-`o_proj` head contributions; pass-1 cache; pass-2 `o_proj.output += alpha * saved`; **two forwards per decode step**
- K=**10** (paper/code for Llama2/Llama3/Olmo)
- Calibration ~25; held-out eval ~50; non-overlapping
"""
)

md("## Setup — packages + HF login + repo path")

code(
    r"""import os, sys, subprocess
from pathlib import Path

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
assert cands, "Attach the repo dataset so common/models.py is under /kaggle/input or /kaggle/working"
REPO = cands[0].parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("REPO", REPO)

import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
print("transformers", transformers.__version__)
assert torch.cuda.device_count() >= 2, "Need T4 x2 for 8B FP16"
"""
)

md(
    """## CELL 1 — Calibration / eval data prep

Load ConflictQA pilot JSONL; split **non-overlapping** calib (25) vs eval (50)."""
)

code(
    r"""from pathlib import Path
import json, random, shutil

from common.conflictqa import load_conflict_items

SEED = 42
N_CALIB = 25
N_EVAL = 50
DATA_NAME = "conflictqa_pilot250_popular.jsonl"
OUT_DIR = Path("/kaggle/working/juice_stage2_fp16")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cands = list(REPO.rglob(DATA_NAME))
cands += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
assert cands, f"Missing {DATA_NAME}"
DATA_PATH = OUT_DIR / DATA_NAME
shutil.copy(cands[0], DATA_PATH)
print("data", cands[0], "->", DATA_PATH)

all_items = load_conflict_items(DATA_PATH)
assert len(all_items) >= N_CALIB + N_EVAL, f"need >= {N_CALIB+N_EVAL} items, got {len(all_items)}"

rng = random.Random(SEED)
ids = [it.id for it in all_items]
rng.shuffle(ids)
calib_ids = set(ids[:N_CALIB])
eval_ids = set(ids[N_CALIB:N_CALIB + N_EVAL])
assert calib_ids.isdisjoint(eval_ids)

calib_items = [it for it in all_items if it.id in calib_ids]
eval_items = [it for it in all_items if it.id in eval_ids]

split = {
    "seed": SEED,
    "n_calib": len(calib_items),
    "n_eval": len(eval_items),
    "calib_ids": sorted(calib_ids),
    "eval_ids": sorted(eval_ids),
}
(OUT_DIR / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
print(f"calib={len(calib_items)} eval={len(eval_items)} overlap={len(calib_ids & eval_ids)}")
print("sample calib", [it.id for it in calib_items[:3]])
print("sample eval", [it.id for it in eval_items[:3]])
"""
)

md(
    """## CELL 2 — Full head identification (real JuICE dual-run, K=10)

Loads Llama-3.1-8B-Instruct FP16 with `device_map="auto"`. Identifies H+/H− for **both** context and parametric targets in one sweep.

**VRAM:** pass-1 `head_out` caches all layers×heads for one sequence (~1–2 GB extra). Batch size 1; cache cleared each item. If OOM, reduce `N_CALIB`."""
)

code(
    r"""from common.models import (
    FP16_8B_MODEL_ID,
    gpu_max_memory,
    load_model_and_tokenizer,
    unload_model,
)
from common.juice import (
    FAST_NEG_ALPHAS,
    FAST_POS_ALPHAS,
    identify_both_directions,
    print_identified,
    save_identified_heads,
)

K = 10  # JuICE: K=10 for Llama2 / Llama3 / Olmo
PRECISION = "fp16"

print("Loading", FP16_8B_MODEL_ID)
model, tokenizer = load_model_and_tokenizer(
    "fp16",
    model_id_fp16=FP16_8B_MODEL_ID,
    device_map="auto",
    max_memory=gpu_max_memory(gpu_fraction=0.90, forbid_cpu_offload=True),
    forbid_cpu_offload=True,
)
print("device_map", getattr(model, "hf_device_map", None))

heads_ctx, heads_par = identify_both_directions(
    model,
    tokenizer,
    calib_items,
    precision=PRECISION,
    K=K,
    pos_alphas=FAST_POS_ALPHAS,   # (1,3,5); paper also uses 10,30
    neg_alphas=FAST_NEG_ALPHAS,   # (-1,-2); paper also uses -5
    seed=SEED,
    checkpoint_dir=OUT_DIR / "id_checkpoints",
)
save_identified_heads(OUT_DIR / "heads_context.json", heads_ctx)
save_identified_heads(OUT_DIR / "heads_parametric.json", heads_par)
print_identified(heads_ctx)
print_identified(heads_par)
print("Saved heads under", OUT_DIR)
"""
)

md(
    """## CELL 3 — Apply steering on held-out eval (baseline / context / parametric)

Three conditions; `follows_context := label == "context"` (corrected scorer with `rejects_context`).

Dual-run generation = **2 forwards per token** — keep `MAX_NEW_TOKENS` modest (16)."""
)

code(
    r"""from common.inference import greedy_generate
from common.juice import apply_juice_steering, context_follow_rate, load_identified_heads
from common.prompts import apply_chat_template
from common.scoring import label_conflict_response
from tqdm import tqdm
import json

MAX_NEW_TOKENS = 16
ALPHA_PLUS = 5.0   # high end of JuICE alpha+ val grid {0..5}
ALPHA_MINUS = 2.0  # mid of JuICE |alpha-| grid {0..3}

heads_ctx = load_identified_heads(OUT_DIR / "heads_context.json")
heads_par = load_identified_heads(OUT_DIR / "heads_parametric.json")

baseline = []
for item in tqdm(eval_items, desc="baseline"):
    prompt = apply_chat_template(tokenizer, item, "context_conflicts")
    text = greedy_generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
    lab = label_conflict_response(text, item.parametric_answer, item.context_answer)
    baseline.append({"id": item.id, "text": text, "label": lab, "follows_context": lab == "context"})

steered_ctx = apply_juice_steering(
    model, tokenizer, eval_items, heads_ctx,
    alpha_plus=ALPHA_PLUS, alpha_minus=ALPHA_MINUS, max_new_tokens=MAX_NEW_TOKENS,
)
steered_par = apply_juice_steering(
    model, tokenizer, eval_items, heads_par,
    alpha_plus=ALPHA_PLUS, alpha_minus=ALPHA_MINUS, max_new_tokens=MAX_NEW_TOKENS,
)

r_base = context_follow_rate(baseline)
r_ctx = context_follow_rate(steered_ctx)
r_par = context_follow_rate(steered_par)

def _counts(rows):
    keys = ("parametric", "context", "rejects_context", "other")
    return {k: sum(1 for r in rows if r["label"] == k) for k in keys}

summary = {
    "n_eval": len(eval_items),
    "follows_context": {
        "baseline": r_base,
        "steer_context": r_ctx,
        "steer_parametric": r_par,
    },
    "delta_pp": {
        "steer_context_minus_baseline": (r_ctx - r_base) * 100,
        "steer_parametric_minus_baseline": (r_par - r_base) * 100,
    },
    "label_counts": {
        "baseline": _counts(baseline),
        "steer_context": _counts(steered_ctx),
        "steer_parametric": _counts(steered_par),
    },
    "alpha_plus": ALPHA_PLUS,
    "alpha_minus": ALPHA_MINUS,
    "K": K,
}
(OUT_DIR / "steer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(OUT_DIR / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
(OUT_DIR / "steer_context.json").write_text(json.dumps(steered_ctx, indent=2), encoding="utf-8")
(OUT_DIR / "steer_parametric.json").write_text(json.dumps(steered_par, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print(
    f"Shift: baseline={r_base:.3f}  +context={r_ctx:.3f} "
    f"(d={(r_ctx-r_base)*100:+.1f} pp)  +parametric={r_par:.3f} "
    f"(d={(r_par-r_base)*100:+.1f} pp)"
)
"""
)

md(
    """## CELL 4 — Sanity / Week-2 gate verdict

JuICE on 7–8B-class models reports strong, controllable shifts. Gate on direction + non-trivial magnitude on `follows_context`, not exact number match."""
)

code(
    r"""import json

summary = json.loads((OUT_DIR / "steer_summary.json").read_text(encoding="utf-8"))
d_ctx = summary["delta_pp"]["steer_context_minus_baseline"]
d_par = summary["delta_pp"]["steer_parametric_minus_baseline"]

ctx_ok = d_ctx >= 8.0
par_ok = d_par <= -8.0
joint_ok = (d_ctx >= 5.0 and d_par <= -5.0) and (ctx_ok or par_ok or (d_ctx - d_par) >= 15.0)

print("follows_context rates:", summary["follows_context"])
print(f"d context steering:    {d_ctx:+.1f} pp  (want clearly positive)")
print(f"d parametric steering: {d_par:+.1f} pp  (want clearly negative)")
print("label_counts:", json.dumps(summary["label_counts"], indent=2))

if (ctx_ok and par_ok) or joint_ok:
    verdict = "JuICE STEERS THIS MODEL (real implementation) — proceed to INT4 comparison"
else:
    verdict = "WEAK/NO EFFECT — investigate before proceeding"

print("\n=== VERDICT ===")
print(verdict)
(OUT_DIR / "verdict.txt").write_text(verdict + "\n", encoding="utf-8")

unload_model(model)
print("Unloaded model.")
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
path = Path(__file__).resolve().parent / "kaggle_stage2_fp16_8b.ipynb"
path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", path, "n_cells", len(cells))
