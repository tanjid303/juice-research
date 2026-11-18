"""Generate juice_repro/kaggle_stage2_cell3_alpha_sweep.ipynb"""
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
    """# Stage 2 — Cell 3-only alpha sweep (reuse saved heads, no re-ID)

**Requires**
1. GPU T4×2, Internet, `HF_TOKEN`, Llama license
2. Dataset with `common/` (same as before)
3. Attach **previous Stage 2 outputs** (or upload): `heads_context.json`, `heads_parametric.json`, `split.json`
4. ConflictQA JSONL (in that pack or the juice16 dataset)

**Does not** re-run head identification.

**Order:** Setup → locate artifacts → head diagnostics → load model → baseline → alpha sweep
"""
)

md("## Setup")

code(
    r"""import os, sys, subprocess, json, re
from pathlib import Path
from collections import Counter

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
assert cands, "Attach dataset containing common/models.py"
REPO = cands[0].parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("REPO", REPO)

import torch
print("cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
assert torch.cuda.device_count() >= 2
"""
)

md(
    """## Locate prior Stage 2 artifacts + rebuild eval split

Searches for `heads_context.json` / `heads_parametric.json` / `split.json` under `/kaggle/input` and `/kaggle/working`.
Upload your downloaded `juice_stage2_fp16` folder as a second dataset if needed.
"""
)

code(
    r"""OUT_DIR = Path("/kaggle/working/juice_stage2_alpha_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    # Prefer prior results folder over anything else
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name} — attach juice_stage2_fp16 outputs dataset"
    # Prefer paths containing juice_stage2
    hits.sort(key=lambda p: (0 if "juice_stage2" in str(p) else 1, len(str(p))))
    print(f"{name} -> {hits[0]}")
    return hits[0]

HEADS_CTX_PATH = find_one("heads_context.json")
HEADS_PAR_PATH = find_one("heads_parametric.json")
SPLIT_PATH = find_one("split.json")

DATA_NAME = "conflictqa_pilot250_popular.jsonl"
data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
assert data_hits, f"Missing {DATA_NAME}"
DATA_PATH = data_hits[0]
print("data ->", DATA_PATH)

from common.conflictqa import load_conflict_items
from common.juice import load_identified_heads, jaccard, HeadRef

split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
eval_ids = set(split["eval_ids"])
all_items = load_conflict_items(DATA_PATH)
eval_items = [it for it in all_items if it.id in eval_ids]
assert len(eval_items) == split["n_eval"], (len(eval_items), split["n_eval"])
print(f"eval_items={len(eval_items)} (from prior split seed={split.get('seed')})")

heads_ctx = load_identified_heads(HEADS_CTX_PATH)
heads_par = load_identified_heads(HEADS_PAR_PATH)
print("loaded heads OK")
"""
)

md(
    """## DIAGNOSTIC — head-set overlap + full ΔP score distributions

Run this **before** trusting any alpha sweep. If enlarge sets are near-identical and scores are uniformly tiny, alpha tuning will not fix ID.
"""
)

code(
    r"""import numpy as np

def head_set(ident, which="enlarge"):
    xs = ident.enlarge if which == "enlarge" else ident.suppress
    return {h.key() for h in xs}

def score_stats(scores: dict, title: str):
    vals = np.array(list(scores.values()), dtype=float)
    print(f"\n--- {title} (n={len(vals)}) ---")
    if len(vals) == 0:
        print("empty")
        return
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    pct = np.percentile(vals, qs)
    for q, v in zip(qs, pct):
        print(f"  p{q:03d}: {v:+.6e}")
    print(f"  mean={vals.mean():+.6e}  std={vals.std():+.6e}")
    print(f"  >0: {(vals > 0).sum()}   <0: {(vals < 0).sum()}   ==0: {(vals == 0).sum()}")
    # top / bottom 10
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    print("  TOP10:")
    for k, s in ranked[:10]:
        print(f"    {k:12s} {s:+.6e}")
    print("  BOTTOM10:")
    for k, s in ranked[-10:]:
        print(f"    {k:12s} {s:+.6e}")

enc_c, enc_p = head_set(heads_ctx, "enlarge"), head_set(heads_par, "enlarge")
sup_c, sup_p = head_set(heads_ctx, "suppress"), head_set(heads_par, "suppress")

def report_overlap(name, a, b):
    inter = a & b
    union = a | b
    jac = len(inter) / len(union) if union else 1.0
    print(f"{name}: |A|={len(a)} |B|={len(b)} |A∩B|={len(inter)} Jaccard={jac:.3f}")
    print(f"  intersection: {sorted(inter)}")
    print(f"  only_context: {sorted(a - b)}")
    print(f"  only_parametric: {sorted(b - a)}")

print("=== Selected-set overlap (context vs parametric) ===")
report_overlap("Enlarge H+", enc_c, enc_p)
report_overlap("Suppress H-", sup_c, sup_p)
# combined intervention sets
comb_c = enc_c | {f"{h.key()}:-" for h in heads_ctx.suppress}  # just for count
print(f"\nRaw enlarge Jaccard (same as above): {jaccard(heads_ctx.enlarge, heads_par.enlarge):.3f}")
print(f"Raw suppress Jaccard: {jaccard(heads_ctx.suppress, heads_par.suppress):.3f}")

score_stats(heads_ctx.scores_pos, "context scores_pos (all heads)")
score_stats(heads_ctx.scores_neg, "context scores_neg (all heads)")
score_stats(heads_par.scores_pos, "parametric scores_pos (all heads)")
score_stats(heads_par.scores_neg, "parametric scores_neg (all heads)")

# Flag for user
jac_e = jaccard(heads_ctx.enlarge, heads_par.enlarge)
top_ctx = max(heads_ctx.scores_pos.values()) if heads_ctx.scores_pos else 0.0
top_par = max(heads_par.scores_pos.values()) if heads_par.scores_pos else 0.0
print("\n=== ID SIGNAL FLAG ===")
if jac_e >= 0.7 and top_ctx < 0.02 and top_par < 0.02:
    print("WARNING: enlarge sets highly overlapping AND peak ΔP tiny.")
    print("Alpha sweep may still be informative for collapse vs coherence,")
    print("but directional steering is unlikely until Cell 2 ID is revisited.")
else:
    print(f"enlarge Jaccard={jac_e:.3f}, peak ΔP ctx={top_ctx:.4f} par={top_par:.4f}")
    print("Proceed with alpha sweep; interpret follows_context only if generations look coherent.")

diag = {
    "enlarge_jaccard": jac_e,
    "suppress_jaccard": jaccard(heads_ctx.suppress, heads_par.suppress),
    "enlarge_intersection": sorted(enc_c & enc_p),
    "peak_score_pos_context": top_ctx,
    "peak_score_pos_parametric": top_par,
}
(OUT_DIR / "head_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
print("wrote", OUT_DIR / "head_diagnostics.json")
"""
)

md(
    """## Load model (FP16, device_map=auto) — no re-ID

Optional: if shards are already cached from the prior session, this is fast.
"""
)

code(
    r"""from common.models import (
    FP16_8B_MODEL_ID,
    gpu_max_memory,
    load_model_and_tokenizer,
    unload_model,
)

print("Loading", FP16_8B_MODEL_ID)
model, tokenizer = load_model_and_tokenizer(
    "fp16",
    model_id_fp16=FP16_8B_MODEL_ID,
    device_map="auto",
    max_memory=gpu_max_memory(gpu_fraction=0.90, forbid_cpu_offload=True),
    forbid_cpu_offload=True,
)
print("device_map", getattr(model, "hf_device_map", None))
"""
)

md(
    """## Baseline on prior eval split (once)

Reuse these rates for every alpha comparison.
"""
)

code(
    r"""from common.inference import greedy_generate
from common.juice import apply_juice_steering, context_follow_rate
from common.prompts import apply_chat_template
from common.scoring import label_conflict_response
from tqdm import tqdm

MAX_NEW_TOKENS = 16
N_SHOW = 5

def label_counts(rows):
    keys = ("parametric", "context", "rejects_context", "other")
    return {k: sum(1 for r in rows if r["label"] == k) for k in keys}

def looks_coherent(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return False
    # classic collapse from prior run
    if t.count("://") >= 3:
        return False
    if re.fullmatch(r"[\W\d_]+", t):
        return False
    # heavy single-token stutter
    toks = t.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters >= max(2, int(0.3 * len(t)))

def coherence_rate(rows) -> float:
    return sum(1 for r in rows if looks_coherent(r["text"])) / max(1, len(rows))

def show_examples(rows, title: str, n: int = N_SHOW):
    print(f"\n===== {title}: {n} raw examples =====")
    for r in rows[:n]:
        print(f"- {r['id']} [{r['label']}] {r['text']!r}")

baseline = []
for item in tqdm(eval_items, desc="baseline"):
    prompt = apply_chat_template(tokenizer, item, "context_conflicts")
    text = greedy_generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
    lab = label_conflict_response(text, item.parametric_answer, item.context_answer)
    baseline.append({"id": item.id, "text": text, "label": lab, "follows_context": lab == "context"})

r_base = context_follow_rate(baseline)
c_base = coherence_rate(baseline)
print(f"BASELINE follows_context={r_base:.3f}  coherence={c_base:.3f}")
print("counts", label_counts(baseline))
show_examples(baseline, "BASELINE")
(OUT_DIR / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
"""
)

md(
    """## Alpha sweep (coherence-gated)

Stages:
1. `(α+=1, α−=0)` enlarge-only  
2. If coherent → `(2, 1)`  
3. If coherent → `(3, 1)`  

Stop early if generations degrade, or if both directions clearly separate from baseline on `follows_context`.
"""
)

code(
    r"""COHERENCE_MIN = 0.70  # fraction of eval texts that must look like English
SEPARATION_PP = 8.0   # clear directional move vs baseline

stages = [
    (1.0, 0.0),
    (2.0, 1.0),
    (3.0, 1.0),
]

sweep_log = []

for alpha_plus, alpha_minus in stages:
    tag = f"ap{alpha_plus}_am{alpha_minus}".replace(".", "p")
    print("\n" + "=" * 70)
    print(f"STAGE alpha_plus={alpha_plus}  alpha_minus={alpha_minus}")
    print("=" * 70)

    steered_ctx = apply_juice_steering(
        model, tokenizer, eval_items, heads_ctx,
        alpha_plus=alpha_plus, alpha_minus=alpha_minus, max_new_tokens=MAX_NEW_TOKENS,
    )
    steered_par = apply_juice_steering(
        model, tokenizer, eval_items, heads_par,
        alpha_plus=alpha_plus, alpha_minus=alpha_minus, max_new_tokens=MAX_NEW_TOKENS,
    )

    r_ctx = context_follow_rate(steered_ctx)
    r_par = context_follow_rate(steered_par)
    coh_ctx = coherence_rate(steered_ctx)
    coh_par = coherence_rate(steered_par)
    d_ctx = (r_ctx - r_base) * 100
    d_par = (r_par - r_base) * 100

    row = {
        "alpha_plus": alpha_plus,
        "alpha_minus": alpha_minus,
        "follows_context": {
            "baseline": r_base,
            "steer_context": r_ctx,
            "steer_parametric": r_par,
        },
        "delta_pp": {
            "steer_context_minus_baseline": d_ctx,
            "steer_parametric_minus_baseline": d_par,
        },
        "coherence": {"baseline": c_base, "steer_context": coh_ctx, "steer_parametric": coh_par},
        "label_counts": {
            "baseline": label_counts(baseline),
            "steer_context": label_counts(steered_ctx),
            "steer_parametric": label_counts(steered_par),
        },
    }
    sweep_log.append(row)

    print(json.dumps(row, indent=2))
    show_examples(steered_ctx, f"STEER CONTEXT α+={alpha_plus} α-={alpha_minus}")
    show_examples(steered_par, f"STEER PARAMETRIC α+={alpha_plus} α-={alpha_minus}")

    (OUT_DIR / f"steer_context_{tag}.json").write_text(json.dumps(steered_ctx, indent=2), encoding="utf-8")
    (OUT_DIR / f"steer_parametric_{tag}.json").write_text(json.dumps(steered_par, indent=2), encoding="utf-8")

    # Coherence gate
    if coh_ctx < COHERENCE_MIN or coh_par < COHERENCE_MIN:
        print(f"\nSTOP: coherence below {COHERENCE_MIN} (ctx={coh_ctx:.2f}, par={coh_par:.2f}). "
              "Do not increase alpha; inspect examples above.")
        break

    # Success / separation gate (both directions)
    sep_ok = (d_ctx >= SEPARATION_PP) and (d_par <= -SEPARATION_PP)
    if sep_ok:
        print(f"\nSTOP: clear bidirectional separation (≥{SEPARATION_PP} pp) at this alpha. "
              "No need to push alpha higher for the gate.")
        break
else:
    print("\nCompleted all planned alpha stages without early stop.")

(OUT_DIR / "alpha_sweep_summary.json").write_text(json.dumps(sweep_log, indent=2), encoding="utf-8")
print("wrote", OUT_DIR / "alpha_sweep_summary.json")

# Final human-readable takeaway
print("\n=== SWEEP TAKEAWAY ===")
for row in sweep_log:
    ap, am = row["alpha_plus"], row["alpha_minus"]
    print(
        f"α+= {ap} α-= {am} | "
        f"fc ctx={row['follows_context']['steer_context']:.3f} "
        f"(d={row['delta_pp']['steer_context_minus_baseline']:+.1f}) | "
        f"fc par={row['follows_context']['steer_parametric']:.3f} "
        f"(d={row['delta_pp']['steer_parametric_minus_baseline']:+.1f}) | "
        f"coh={row['coherence']['steer_context']:.2f}/{row['coherence']['steer_parametric']:.2f}"
    )

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
path = Path(__file__).resolve().parent / "kaggle_stage2_cell3_alpha_sweep.ipynb"
path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", path, "cells", len(cells))
