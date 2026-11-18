"""Generate juice_repro/kaggle_stage2_int4_transfer.ipynb"""
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
    """# Stage 2 — FP16→INT4 transfer (base, no re-ID)

Apply **FP16-identified** base heads to the **self-quantized GPTQ-INT4** Llama-3.1-8B base.

**Identify on:** FP16 base (`juice_stage2_base_id/directional_heads/`)  
**Apply to:** `/kaggle/working/llama31_8b_base_gptq_int4` (or attached notebook output)  
**Eval:** same 50 held-out items (`split.json`)  
**Prompts:** JuICE `coherent_conflict` (`juice_coherent`)  
**Gain:** `alpha_plus=1`, `alpha_minus=0`

| Condition | Heads | FP16 base ref |
|-----------|-------|----------------|
| baseline | none | fc 16% |
| steer_context_k10 | directional ctx top-10 | **+10 pp** |
| steer_context_k5 | directional ctx top-5 | **+6 pp** |
| steer_parametric_k2_clean | L20_H14, L26_H15 | **−8 pp** |
| steer_parametric_k2 | L20_H14, L31_H14 | −6 pp |
| steer_parametric_k3 | + L26_H15 | −6 pp |

Transfer **passes** if INT4 deltas keep the same sign as FP16 and coherence stays high (not a magnitude match).

## Before Run All

1. **GPU T4 ×2**, Internet ON
2. Attach **this new zip** as a dataset (not the old `1234567890` code zip)
3. Attach **quantizing-output** (notebook output with `llama31_8b_base_gptq_int4/`)
4. **File → Upload notebook** using `juice_repro/kaggle_stage2_int4_transfer.ipynb` from the new zip
5. Save & Run All (fresh kernel — do not reuse the crashed session)
"""
)

md("## 1 — Setup")
code(
    r"""import os, sys, subprocess, json, re
from pathlib import Path
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==4.51.3",
        "accelerate>=0.26.0,<2.0",
        "safetensors>=0.4.0",
        "gptqmodel==2.2.0",
        "huggingface-hub>=0.26.0",
        "tqdm",
    ]
)

import torch
import transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "ngpu", torch.cuda.device_count())
assert torch.cuda.is_available()

cands = list(Path("/kaggle/working").rglob("common/models.py"))
cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
assert cands, "Attach the INT4 transfer dataset (must contain common/)"

def _juice_attn_ok(models_py: Path) -> bool:
    ja = models_py.parent / "juice_attn.py"
    if not ja.is_file():
        return False
    return "dense_linear_weight_T" in ja.read_text(encoding="utf-8", errors="ignore")

cands.sort(key=lambda p: (_juice_attn_ok(p), "int4" in str(p).lower()), reverse=True)
assert _juice_attn_ok(cands[0]), (
    "This session still has the OLD common/ (no GPTQ o_proj dequant). "
    "Add the NEW kaggle_juice_stage2_int4_transfer.zip as a dataset and remove the old 1234567890 input."
)
REPO = cands[0].parent.parent
# Copy into /kaggle/working so this kernel cannot silently import the old zip.
import shutil
dst_common = Path("/kaggle/working/common")
if dst_common.exists():
    shutil.rmtree(dst_common)
shutil.copytree(REPO / "common", dst_common)
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("REPO", REPO)
from common.juice_attn import dense_linear_weight_T  # noqa: F401
print("juice_attn GPTQ o_proj dequant: OK")

try:
    from huggingface_hub import login
    from kaggle_secrets import UserSecretsClient
    login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
    print("HF login ok")
except Exception as e:
    print("HF login skipped:", type(e).__name__, e)
"""
)

md("## 2 — Locate FP16 heads, eval split, INT4 checkpoint")
code(
    r"""OUT_DIR = Path("/kaggle/working/juice_stage2_int4_transfer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paste the folder that contains config.json + *.safetensors after you attach
# the quantize notebook output. Leave None to auto-detect.
INT4_MODEL_ID_OVERRIDE = None
# Example after Add Data → Notebook Output Files:
# INT4_MODEL_ID_OVERRIDE = "/kaggle/input/quantizing-output/llama31_8b_base_gptq_int4"

print("=== /kaggle/input (top-level) ===")
inp = Path("/kaggle/input")
if inp.exists():
    for p in sorted(inp.iterdir()):
        print(" ", "DIR " if p.is_dir() else "FILE", p)
else:
    print("  MISSING — attach datasets in the notebook sidebar")

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
assert data_hits, "Missing conflictqa_pilot250_popular.jsonl"
DATA_PATH = data_hits[0]
print("DATA", DATA_PATH)

from common.conflictqa import load_conflict_items
from common.juice import load_identified_heads
from common.models import resolve_gptq_8b_base_model_id

INT4_MODEL_ID = resolve_gptq_8b_base_model_id(INT4_MODEL_ID_OVERRIDE)
print("INT4_MODEL_ID", INT4_MODEL_ID)

split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
eval_ids = set(split["eval_ids"])
eval_items = [it for it in load_conflict_items(DATA_PATH) if it.id in eval_ids]
assert len(eval_items) == split["n_eval"], (len(eval_items), split["n_eval"])
print("n_eval", len(eval_items), "seed", split.get("seed"))

for tag, path in HEADS.items():
    ident = load_identified_heads(path)
    print(tag, "->", [h.key() for h in ident.enlarge])
"""
)

md("## 3 — Load INT4 base (GPTQ, GPU 0)")
code(
    r"""from common.models import load_model_and_tokenizer

print("Loading INT4", INT4_MODEL_ID)
model, tokenizer = load_model_and_tokenizer(
    "int4",
    model_id_int4=INT4_MODEL_ID,
    device_map={"": 0},
)
print("device_map", getattr(model, "hf_device_map", None))
print("config.quantization_config", getattr(model.config, "quantization_config", None))
from common.juice import wrap_llama_attention_for_juice
wrap_llama_attention_for_juice(model)
print("wrapped JuICE attention (GPTQ o_proj dequant cached)")
"""
)

md("## 4 — Transfer conditions (same protocol as FP16 Cell-3)")
code(
    r"""from common.inference import greedy_generate
from common.juice import apply_juice_steering, context_follow_rate, load_identified_heads, conflict_eval_prompt
from common.scoring import label_conflict_response
from common.models import unload_model
from tqdm import tqdm

MAX_NEW_TOKENS = 16
ALPHA_PLUS = 1.0
ALPHA_MINUS = 0.0
N_SHOW = 5
PROMPT_FORMAT = "juice_coherent"

FP16_REF = {
    "baseline_fc": 0.16,
    "steer_context_k10": 10.0,
    "steer_context_k5": 6.0,
    "steer_parametric_k2": -6.0,
    "steer_parametric_k3": -6.0,
    "steer_parametric_k2_clean": -8.0,
    "coherence": 1.0,
    "note": "FP16 base Cell-3, juice_coherent, alpha+=1 alpha-=0, n=50",
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

baseline = []
for item in tqdm(eval_items, desc="baseline"):
    prompt = conflict_eval_prompt(tokenizer, item, prompt_format=PROMPT_FORMAT)
    text = greedy_generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
    lab = label_conflict_response(text, item.parametric_answer, item.context_answer)
    baseline.append({"id": item.id, "text": text, "label": lab, "follows_context": lab == "context"})

r_base = context_follow_rate(baseline)
c_base = coherence_rate(baseline)
print(f"INT4 baseline follows_context={r_base:.3f} coherence={c_base:.3f} counts={label_counts(baseline)}")
print(f"FP16 ref baseline fc={FP16_REF['baseline_fc']:.3f}")
show_examples(baseline, "INT4 BASELINE (juice_coherent)")
(OUT_DIR / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

conditions = [
    ("steer_context_k10", HEADS["ctx_k10"]),
    ("steer_context_k5", HEADS["ctx_k5"]),
    ("steer_parametric_k2_clean", HEADS["par_k2_clean"]),
    ("steer_parametric_k2", HEADS["par_k2"]),
    ("steer_parametric_k3", HEADS["par_k3"]),
]

summary = {
    "cell": "fp16_to_int4_transfer",
    "identify_on": "fp16_base",
    "apply_to": "int4_base",
    "model": INT4_MODEL_ID,
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
    "fp16_reference": FP16_REF,
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
        "fp16_delta_pp": FP16_REF.get(tag),
        "same_sign_as_fp16": (
            (d_pp == 0 and FP16_REF.get(tag, 0) == 0)
            or (d_pp * FP16_REF.get(tag, 0) > 0)
        ),
    }
    summary["conditions"][tag] = entry
    print(json.dumps(entry, indent=2))
    show_examples(rows, tag)
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

gate_tags = ["steer_context_k10", "steer_context_k5", "steer_parametric_k2_clean"]
gate_ok = all(summary["conditions"][t]["same_sign_as_fp16"] for t in gate_tags)
gate_coh = all(summary["conditions"][t]["coherence"] >= 0.90 for t in gate_tags) and c_base >= 0.90
summary["transfer_gate"] = {
    "primary_conditions": gate_tags,
    "same_sign": gate_ok,
    "coherence_ok": gate_coh,
    "passed": bool(gate_ok and gate_coh),
}

(OUT_DIR / "int4_transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n=== TAKEAWAY TABLE (INT4 transfer vs FP16) ===")
print(f"{'condition':32s} {'int4_fc':>8s} {'int4_d':>8s} {'fp16_d':>8s} {'sign':>6s} {'coh':>6s}")
print(f"{'baseline':32s} {r_base:8.3f} {0.0:8.1f} {0.0:8.1f} {'':6s} {c_base:6.3f}")
for tag, entry in summary["conditions"].items():
    sign = "OK" if entry["same_sign_as_fp16"] else "FLIP"
    print(
        f"{tag:32s} {entry['follows_context']:8.3f} "
        f"{entry['delta_pp_vs_baseline']:8.1f} {entry['fp16_delta_pp']:8.1f} "
        f"{sign:6s} {entry['coherence']:6.3f}"
    )

print("\n=== TRANSFER GATE (ctx K10/K5 + parametric clean K2) ===")
print(json.dumps(summary["transfer_gate"], indent=2))
if summary["transfer_gate"]["passed"]:
    print("GATE PASSED: FP16 heads move INT4 in the same direction; coherence held.")
else:
    print("GATE FAILED: inspect sign flips / coherence before native INT4 ID.")

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
out = Path(__file__).resolve().parent / "kaggle_stage2_int4_transfer.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
