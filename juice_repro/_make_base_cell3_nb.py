"""Generate juice_repro/kaggle_stage2_base_cell3_directional.ipynb"""
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
    """# Stage 2 — Base Cell 3 directional steering (no re-ID)

**Model:** `meta-llama/Llama-3.1-8B` (base, not Instruct)  
**Eval:** same 50 held-out items as Instruct (`split.json`)  
**Prompts:** JuICE `coherent_conflict` plain completion (no chat template)  
**Steering:** `alpha_plus=1`, `alpha_minus=0` (enlarge-only)

| Condition | Heads |
|-----------|-------|
| baseline | none |
| steer_context_k10 | directional ctx top-10 |
| steer_context_k5 | directional ctx top-5 |
| steer_parametric_k2 | directional par top-2 (L20_H14, L31_H14) |
| steer_parametric_k3 | directional par top-3 |
| steer_parametric_k2_clean | L20_H14, L26_H15 (positive par_pos; less context-suppression) |

Compare vs Instruct directional run (~28% baseline fc, +6–8pp context, 0pp parametric).
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

md("## Locate artifacts + eval split (same 50 as Instruct)")
code(
    r"""OUT_DIR = Path("/kaggle/working/juice_stage2_base_directional_cell3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if "directional" in str(p) or "base_id" in str(p) else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

HEADS = {
    "ctx_k10": find_one("heads_dir_context_k10.json"),
    "ctx_k5": find_one("heads_dir_context_k5.json"),
    "par_k2": find_one("heads_dir_parametric_k2.json"),
    "par_k3": find_one("heads_dir_parametric_k3.json"),
    "par_k2_clean": find_one("heads_dir_parametric_k2_clean.json"),
}
SPLIT_PATH = find_one("split.json")
DATA_NAME = "conflictqa_pilot250_popular.jsonl"
data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
assert data_hits
DATA_PATH = data_hits[0]

from common.conflictqa import load_conflict_items
from common.juice import load_identified_heads

split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
eval_ids = set(split["eval_ids"])
eval_items = [it for it in load_conflict_items(DATA_PATH) if it.id in eval_ids]
assert len(eval_items) == split["n_eval"], (len(eval_items), split["n_eval"])
print("n_eval", len(eval_items), "seed", split.get("seed"))

for tag, path in HEADS.items():
    ident = load_identified_heads(path)
    print(tag, "->", [h.key() for h in ident.enlarge], ident.selection_method if hasattr(ident, 'selection_method') else path.name)
"""
)

md("## Load base 8B FP16")
code(
    r"""from common.models import FP16_8B_BASE_MODEL_ID, gpu_max_memory, load_model_and_tokenizer, unload_model

print("Loading", FP16_8B_BASE_MODEL_ID)
model, tokenizer = load_model_and_tokenizer(
    "fp16",
    model_id_fp16=FP16_8B_BASE_MODEL_ID,
    device_map="auto",
    max_memory=gpu_max_memory(gpu_fraction=0.90, forbid_cpu_offload=True),
    forbid_cpu_offload=True,
)
print("device_map", getattr(model, "hf_device_map", None))
"""
)

md("## Run conditions (JuICE coherent_conflict prompts, α+=1, α−=0)")
code(
    r"""from common.inference import greedy_generate
from common.juice import apply_juice_steering, context_follow_rate, load_identified_heads, conflict_eval_prompt
from common.scoring import label_conflict_response
from tqdm import tqdm

MAX_NEW_TOKENS = 16
ALPHA_PLUS = 1.0
ALPHA_MINUS = 0.0
N_SHOW = 5
PROMPT_FORMAT = "juice_coherent"

INSTRUCT_REF = {
    "baseline_fc": 0.28,
    "steer_context_k10_fc_delta": 8.0,
    "steer_context_k5_fc_delta": 6.0,
    "steer_parametric_k2_fc_delta": 0.0,
    "steer_parametric_k3_fc_delta": -2.0,
    "note": "Instruct used chat template; parametric K=2/3 did not move fc meaningfully",
}

def label_counts(rows):
    keys = ("parametric", "context", "rejects_context", "other")
    return {k: sum(1 for r in rows if r["label"] == k) for k in keys}

def looks_coherent(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return False
    if t.count("://") >= 3:
        return False
    if re.fullmatch(r"[\W\d_]+", t):
        return False
    toks = t.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters >= max(2, int(0.3 * len(t)))

def coherence_rate(rows):
    return sum(1 for r in rows if looks_coherent(r["text"])) / max(1, len(rows))

def show_examples(rows, title):
    print(f"\n===== {title}: {N_SHOW} raw examples =====")
    for r in rows[:N_SHOW]:
        print(f"- {r['id']} [{r['label']}] {r['text']!r}")

# Baseline (plain JuICE conflict prompt, no steering)
baseline = []
for item in tqdm(eval_items, desc="baseline"):
    prompt = conflict_eval_prompt(tokenizer, item, prompt_format=PROMPT_FORMAT)
    text = greedy_generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
    lab = label_conflict_response(text, item.parametric_answer, item.context_answer)
    baseline.append({"id": item.id, "text": text, "label": lab, "follows_context": lab == "context"})

r_base = context_follow_rate(baseline)
c_base = coherence_rate(baseline)
print(f"BASE baseline follows_context={r_base:.3f} coherence={c_base:.3f} counts={label_counts(baseline)}")
show_examples(baseline, "BASE BASELINE (juice_coherent)")
(OUT_DIR / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

conditions = [
    ("steer_context_k10", HEADS["ctx_k10"]),
    ("steer_context_k5", HEADS["ctx_k5"]),
    ("steer_parametric_k2", HEADS["par_k2"]),
    ("steer_parametric_k3", HEADS["par_k3"]),
    ("steer_parametric_k2_clean", HEADS["par_k2_clean"]),
]

summary = {
    "model": FP16_8B_BASE_MODEL_ID,
    "prompt_format": PROMPT_FORMAT,
    "alpha_plus": ALPHA_PLUS,
    "alpha_minus": ALPHA_MINUS,
    "n_eval": len(eval_items),
    "baseline": {
        "follows_context": r_base,
        "coherence": c_base,
        "label_counts": label_counts(baseline),
    },
    "conditions": {},
    "instruct_reference": INSTRUCT_REF,
}

for tag, path in conditions:
    ident = load_identified_heads(path)
    print("\n" + "=" * 70)
    print(tag, "heads", [h.key() for h in ident.enlarge])
    print("=" * 70)
    rows = apply_juice_steering(
        model,
        tokenizer,
        eval_items,
        ident,
        alpha_plus=ALPHA_PLUS,
        alpha_minus=ALPHA_MINUS,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_format=PROMPT_FORMAT,
    )
    r = context_follow_rate(rows)
    coh = coherence_rate(rows)
    d_pp = (r - r_base) * 100
    entry = {
        "heads": [h.key() for h in ident.enlarge],
        "steer_target": ident.steer_target,
        "follows_context": r,
        "delta_pp_vs_baseline": d_pp,
        "coherence": coh,
        "label_counts": label_counts(rows),
    }
    summary["conditions"][tag] = entry
    print(json.dumps(entry, indent=2))
    show_examples(rows, tag)
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

(OUT_DIR / "base_directional_cell3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n=== TAKEAWAY TABLE (BASE) ===")
print(f"{'condition':32s} {'fc':>6s} {'d_pp':>7s} {'coh':>6s}")
print(f"{'baseline':32s} {r_base:6.3f} {0.0:7.1f} {c_base:6.3f}")
for tag, entry in summary["conditions"].items():
    print(
        f"{tag:32s} {entry['follows_context']:6.3f} "
        f"{entry['delta_pp_vs_baseline']:7.1f} {entry['coherence']:6.3f}"
    )

print("\n=== vs INSTRUCT (reference deltas on fc) ===")
for tag in ["steer_context_k10", "steer_context_k5", "steer_parametric_k2", "steer_parametric_k3"]:
    ref_key = tag + "_fc_delta"
    if ref_key in INSTRUCT_REF and tag in summary["conditions"]:
        base_d = summary["conditions"][tag]["delta_pp_vs_baseline"]
        print(f"  {tag}: BASE {base_d:+.1f} pp  |  Instruct ref {INSTRUCT_REF[ref_key]:+.1f} pp")

print("\n=== PARAMETRIC READ ===")
for name in ["steer_parametric_k2", "steer_parametric_k3", "steer_parametric_k2_clean"]:
    e = summary["conditions"][name]
    moved_down = e["delta_pp_vs_baseline"] <= -5.0
    moved_up = e["delta_pp_vs_baseline"] >= 5.0
    verdict = "MOVED toward parametric (fc down)" if moved_down else (
        "MOVED toward context (fc up)" if moved_up else "NO clear directional move"
    )
    print(
        f"  {name}: d_fc={e['delta_pp_vs_baseline']:+.1f} pp, coh={e['coherence']:.2f}, "
        f"P/C/O={e['label_counts']['parametric']}/{e['label_counts']['context']}/{e['label_counts']['other']} -> {verdict}"
    )

k2 = summary["conditions"]["steer_parametric_k2"]
k2c = summary["conditions"]["steer_parametric_k2_clean"]
print("\n=== CLEAN vs FULL parametric K=2 ===")
print(f"  full (L20_H14,L31_H14): d_fc={k2['delta_pp_vs_baseline']:+.1f} pp, coh={k2['coherence']:.2f}")
print(f"  clean (L20_H14,L26_H15): d_fc={k2c['delta_pp_vs_baseline']:+.1f} pp, coh={k2c['coherence']:.2f}")
if abs(k2c["delta_pp_vs_baseline"]) > abs(k2["delta_pp_vs_baseline"]) + 2:
    print("  -> CLEAN heads show STRONGER parametric-directional behavioral effect")
elif k2["coherence"] < 0.9 and k2c["coherence"] >= 0.9:
    print("  -> FULL set noisier; CLEAN heads more coherent")
else:
    print("  -> Similar behavior; inspect raw examples above")

unload_model(model)
print("done", OUT_DIR)
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
path = Path(__file__).resolve().parent / "kaggle_stage2_base_cell3_directional.ipynb"
path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", path)
