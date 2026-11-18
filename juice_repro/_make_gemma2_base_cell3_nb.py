"""Generate juice_repro/kaggle_gemma2_base_cell3.ipynb — FP16 Cell-3 apply on Gemma-2-9B base."""
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
    """# Gemma-2-9B base — Cell 3 FP16 apply (no GPTQ)

Copy of the Llama Stage 2 **base Cell-3** protocol on **`google/gemma-2-9b`**.

Does **not** load or apply the INT4 checkpoint. Report this FP16 gate before any GPTQ transfer.

| Setting | Value |
|---------|--------|
| Model | `google/gemma-2-9b` **base** |
| Eval | same 50 IDs as Gemma ID (`split.json`, seed 42) |
| Prompt | JuICE `coherent_conflict` (`juice_coherent`, no chat template) |
| Decode | greedy, `max_new_tokens=16` |
| α | α+=**1**, α−=**0** (enlarge-only) |
| Bootstrap | paired, 1000 draws, seed 42 |
| Attn | eager (`GetHeadsGemma2Attention`) |

| Condition | Heads | Role |
|-----------|-------|------|
| baseline | none | — |
| steer_context_k10 | directional ctx top-10 | **headline** |
| steer_parametric_k2_clean | L31_H7, L39_H6 | protocol pair (directional = clean; expected suppressor) |
| steer_parametric_k1_raw | L27_H6 | methods check (Llama L1_H4 analog) |
| random_context_k10 | 10 heads, seed 42, exclude identified set | same-α control |

**~1.5–3 h** on GPU T4 ×2. Run All in the **draft**. Quick Save when it prints `SAVE VERSION NOW`.
"""
)

md("## Setup")
code(
    r"""import os, sys, subprocess, json, re, time
from pathlib import Path

STAGE_T0 = time.time()

def elapsed(stage: str) -> None:
    print(f"[elapsed] {stage}: {(time.time() - STAGE_T0) / 60:.1f} min since start")

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
        "numpy",
        "pandas",
    ]
)

from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)

cands = list(Path("/kaggle/working").rglob("common/models.py"))
cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
assert cands, "Attach kaggle_juice_gemma2_base_cell3.zip"

def _juice_attn_ok(models_py: Path) -> bool:
    ja = models_py.parent / "juice_attn.py"
    if not ja.is_file():
        return False
    txt = ja.read_text(encoding="utf-8", errors="ignore")
    return "GetHeadsGemma2Attention" in txt

cands.sort(key=lambda p: (_juice_attn_ok(p), "gemma2" in str(p).lower(), "cell3" in str(p).lower()), reverse=True)
assert _juice_attn_ok(cands[0]), "common/juice_attn.py missing GetHeadsGemma2Attention"
REPO = cands[0].parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("REPO", REPO)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
assert torch.cuda.is_available()
if torch.cuda.device_count() < 2:
    print("WARNING: prefer T4 x2")
elapsed("setup")
"""
)

md("## Locate Gemma ID heads + locked 50 eval IDs")
code(
    r"""OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_base_cell3")
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

HEADS = {
    "ctx_k10": find_one("heads_dir_context_k10.json"),
    "par_k2_clean": find_one("heads_dir_parametric_k2_clean.json"),
    "par_k2": find_one("heads_dir_parametric_k2.json"),
    "par_k1_raw": find_one("heads_dir_parametric_k1_raw.json"),
}
SPLIT_PATH = find_one("split.json")
DATA_NAME = "conflictqa_pilot250_popular.jsonl"
data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
assert data_hits, DATA_NAME
DATA_PATH = data_hits[0]

from common.conflictqa import load_conflict_items
from common.juice import load_identified_heads, head_keys

split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
eval_ids = set(split["eval_ids"])
eval_items = [it for it in load_conflict_items(DATA_PATH) if it.id in eval_ids]
assert len(eval_items) == split["n_eval"] == 50, (len(eval_items), split["n_eval"])
print("n_eval", len(eval_items), "seed", split.get("seed"))

ident_ctx = load_identified_heads(HEADS["ctx_k10"])
ident_par = load_identified_heads(HEADS["par_k2_clean"])
ident_par_dir = load_identified_heads(HEADS["par_k2"])
ident_raw = load_identified_heads(HEADS["par_k1_raw"])
assert head_keys(ident_par.enlarge) == head_keys(ident_par_dir.enlarge) == ["L31_H7", "L39_H6"]
assert head_keys(ident_raw.enlarge) == ["L27_H6"]
print("ctx_k10", head_keys(ident_ctx.enlarge))
print("par_k2 (directional=clean)", head_keys(ident_par.enlarge))
print("par_k1_raw", head_keys(ident_raw.enlarge))
elapsed("artifacts")
"""
)

md("## Load Gemma-2-9B FP16")
code(
    r"""from common.models import gpu_max_memory, load_model_and_tokenizer, unload_model

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
assert model.config._attn_implementation == "eager"
elapsed("model loaded")
"""
)

md("## Run conditions (juice_coherent, α+=1, α−=0, paired bootstrap)")
code(
    r"""from common.inference import greedy_generate
from common.juice import (
    HeadRef,
    apply_juice_steering,
    conflict_eval_prompt,
    context_follow_rate,
    identified_from_enlarge,
    model_dims,
    random_head_control,
)
from common.scoring import label_conflict_response
from common.stats_analysis import bootstrap_paired_follow_delta
from tqdm import tqdm

MAX_NEW_TOKENS = 16
ALPHA_PLUS = 1.0
ALPHA_MINUS = 0.0
N_SHOW = 5
PROMPT_FORMAT = "juice_coherent"
N_BOOT = 1000
BOOT_SEED = 42
RAND_SEED = 42

def label_counts(rows):
    keys = ("parametric", "context", "rejects_context", "other")
    return {k: sum(1 for r in rows if r["label"] == k) for k in keys}

def pco(counts):
    return f"{counts['parametric']}/{counts['context']}/{counts['other']}"

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

def fmt_ci(boot):
    d = boot["delta"]
    lo, hi = d["ci95_pp"]
    return f"[{lo:+.1f}, {hi:+.1f}]"

# --- baseline ---
baseline = []
for item in tqdm(eval_items, desc="baseline"):
    prompt = conflict_eval_prompt(tokenizer, item, prompt_format=PROMPT_FORMAT)
    text = greedy_generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
    lab = label_conflict_response(text, item.parametric_answer, item.context_answer)
    baseline.append({"id": item.id, "text": text, "label": lab, "follows_context": lab == "context"})

r_base = context_follow_rate(baseline)
c_base = coherence_rate(baseline)
print(f"BASELINE fc={r_base:.3f} coh={c_base:.3f} P/C/O={pco(label_counts(baseline))} counts={label_counts(baseline)}")
show_examples(baseline, "BASELINE (juice_coherent)")
(OUT_DIR / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
elapsed("baseline")

n_layers, n_heads, head_dim = model_dims(model)
print("dims", n_layers, n_heads, head_dim)

identified_exclude = list(ident_ctx.enlarge) + list(ident_par.enlarge) + list(ident_raw.enlarge)
rand_heads = random_head_control(n_layers, n_heads, 10, seed=RAND_SEED, exclude=identified_exclude)
ident_rand = identified_from_enlarge(
    rand_heads, precision="fp16_gemma2_random", K=10, steer_target="context"
)
overlap = set(head_keys(ident_rand.enlarge)) & set(head_keys(identified_exclude))
assert not overlap, overlap
print("random_k10", head_keys(ident_rand.enlarge), "excluded", head_keys(identified_exclude))
(OUT_DIR / "random_k10_heads.json").write_text(
    json.dumps({"heads": head_keys(ident_rand.enlarge), "seed": RAND_SEED, "excluded": head_keys(identified_exclude)}, indent=2),
    encoding="utf-8",
)

conditions = [
    ("steer_context_k10", ident_ctx),
    ("steer_parametric_k2_clean", ident_par),
    ("steer_parametric_k1_raw", ident_raw),
    ("random_context_k10", ident_rand),
]

summary = {
    "model": MODEL_ID,
    "prompt_format": PROMPT_FORMAT,
    "alpha_plus": ALPHA_PLUS,
    "alpha_minus": ALPHA_MINUS,
    "n_eval": len(eval_items),
    "split_seed": split.get("seed"),
    "n_boot": N_BOOT,
    "boot_seed": BOOT_SEED,
    "random_k_seed": RAND_SEED,
    "baseline": {
        "follows_context": r_base,
        "coherence": c_base,
        "label_counts": label_counts(baseline),
    },
    "conditions": {},
}

row_store = {}
for tag, ident in conditions:
    print("\n" + "=" * 70)
    print(tag, "heads", head_keys(ident.enlarge), "target", ident.steer_target)
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
    boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=N_BOOT, seed=BOOT_SEED)
    entry = {
        "heads": head_keys(ident.enlarge),
        "steer_target": ident.steer_target,
        "K": ident.K,
        "follows_context": r,
        "delta_pp_vs_baseline": d_pp,
        "coherence": coh,
        "label_counts": label_counts(rows),
        "bootstrap": boot["delta"],
    }
    summary["conditions"][tag] = entry
    row_store[tag] = rows
    print(json.dumps({k: v for k, v in entry.items() if k != "bootstrap"}, indent=2))
    print(
        "bootstrap delta_pp",
        boot["delta"]["point_pp"],
        "CI",
        boot["delta"]["ci95_pp"],
        "excl0",
        boot["delta"]["excludes_zero"],
    )
    show_examples(rows, tag)
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elapsed(tag)

(OUT_DIR / "gemma2_base_cell3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

NOTE = (
    "Resample the 50 eval items with replacement. Delta = steered "
    "follows_context − baseline follows_context, paired on item id."
)
boot_payload = {
    "n_boot": N_BOOT,
    "seed": BOOT_SEED,
    "note": NOTE,
    "model": MODEL_ID,
    "precision": "fp16",
    "primary_conditions": [
        "steer_context_k10",
        "steer_parametric_k2_clean",
        "steer_parametric_k1_raw",
        "random_context_k10",
    ],
    "fp16": {
        "precision": "fp16",
        "n_baseline": len(baseline),
        "conditions": {
            tag: bootstrap_paired_follow_delta(baseline, row_store[tag], n_boot=N_BOOT, seed=BOOT_SEED)
            for tag in summary["conditions"]
        },
    },
}
(OUT_DIR / "gemma2_base_cell3_bootstrap.json").write_text(json.dumps(boot_payload, indent=2), encoding="utf-8")

print("\n=== GEMMA-2-9B FP16 GATE TABLE ===")
hdr = f"{'condition':28s} {'fc':>6s} {'d_pp':>7s} {'CI95':>18s} {'excl0':>6s} {'coh':>6s} {'P/C/O':>8s}"
print(hdr)
print("-" * len(hdr))
print(f"{'baseline':28s} {r_base:6.3f} {0.0:7.1f} {'—':>18s} {'—':>6s} {c_base:6.3f} {pco(label_counts(baseline)):>8s}")
for tag, entry in summary["conditions"].items():
    d = entry["bootstrap"]
    lo, hi = d["ci95_pp"]
    ci = f"[{lo:+.1f}, {hi:+.1f}]"
    print(
        f"{tag:28s} {entry['follows_context']:6.3f} "
        f"{entry['delta_pp_vs_baseline']:7.1f} {ci:>18s} "
        f"{str(d['excludes_zero']):>6s} {entry['coherence']:6.3f} "
        f"{pco(entry['label_counts']):>8s}"
    )

print("\n=== READ ===")
ctx = summary["conditions"]["steer_context_k10"]
par = summary["conditions"]["steer_parametric_k2_clean"]
raw = summary["conditions"]["steer_parametric_k1_raw"]
rnd = summary["conditions"]["random_context_k10"]
print(
    f"  headline ctx K10: d={ctx['delta_pp_vs_baseline']:+.1f} pp "
    f"CI={fmt_ci(boot_payload['fp16']['conditions']['steer_context_k10'])} "
    f"excl0={ctx['bootstrap']['excludes_zero']} P/C/O={pco(ctx['label_counts'])}"
)
print(
    f"  protocol par K2 (L31_H7, L39_H6): d={par['delta_pp_vs_baseline']:+.1f} pp "
    f"CI={fmt_ci(boot_payload['fp16']['conditions']['steer_parametric_k2_clean'])} "
    f"excl0={par['bootstrap']['excludes_zero']} P/C/O={pco(par['label_counts'])}"
)
print(
    f"  raw par peak L27_H6 K=1: d={raw['delta_pp_vs_baseline']:+.1f} pp "
    f"CI={fmt_ci(boot_payload['fp16']['conditions']['steer_parametric_k1_raw'])} "
    f"excl0={raw['bootstrap']['excludes_zero']} P/C/O={pco(raw['label_counts'])}"
)
print(
    f"  random-K10: d={rnd['delta_pp_vs_baseline']:+.1f} pp "
    f"CI={fmt_ci(boot_payload['fp16']['conditions']['random_context_k10'])} "
    f"excl0={rnd['bootstrap']['excludes_zero']} P/C/O={pco(rnd['label_counts'])}"
)
print("Do not start GPTQ apply/transfer until this table is copied back.")

unload_model(model)
print()
print("=" * 60)
print("SAVE VERSION NOW — Quick Save, not Save & Run All")
print("Need: gemma2_base_cell3_summary.json, gemma2_base_cell3_bootstrap.json")
print("folder:", OUT_DIR)
print("=" * 60)
elapsed("done")
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
out = Path(__file__).resolve().parent / "kaggle_gemma2_base_cell3.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("Wrote", out)
