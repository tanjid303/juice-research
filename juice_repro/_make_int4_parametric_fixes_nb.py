"""Generate juice_repro/kaggle_stage2_int4_parametric_fixes.ipynb

2. directional_par ranks 1-20 (from saved INT4 ID scores)
4. L1_H4 alone (K=1) as parametric enlarge on INT4
1. Alpha grid on FP16 clean pair → INT4
3. Suppress FP16 context K10 heads (H−) as a parametric push
"""
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
    """# Stage 2 — INT4 parametric recovery checks (fast)

Do **not** re-run native ID. Reuses FP16 heads + saved INT4 score maps.

| # | Check | GPU? |
|---|--------|------|
| 2 | directional_par ranks 1–20 + positive `par_pos` scan | no |
| **4** | **L1_H4 alone (K=1), α+=1** — only INT4 head with meaningful +par_pos | yes (run first after baseline) |
| 1 | Alpha grid on **L20_H14, L26_H15**: 0.5 / 1 / 2 / 3 / 5 | yes |
| 3 | Suppress context K10 heads (H−, α=1) as parametric push | yes |

L1_H4 is **not** an INT4-only discovery: it was FP16 **raw parametric enlarge #1** (`par_pos` 0.0577), skipped for Cell-3 because directional rerank ranked it #29. INT4 `par_pos` 0.0578 (still raw #1). Same-head story if it steers.

Existing transfer: clean K2 at α=1 was **−2 pp**, CI [−6, 0], coherence 100%.

## Before Run All

1. GPU **T4 ×2**, Internet ON
2. Attach this zip **and** `quantizing-output`
3. Save & Run All (~40–70 min: baseline + L1_H4 + 5 alphas + 1 suppress × 50 items)
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
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.51.3",
        "accelerate>=0.26.0,<2.0",
        "safetensors>=0.4.0",
        "gptqmodel==2.2.0",
        "huggingface-hub>=0.26.0",
        "tqdm",
    ]
)

import torch, transformers, shutil
print("torch", torch.__version__, "transformers", transformers.__version__, "ngpu", torch.cuda.device_count())
assert torch.cuda.is_available()

cands = list(Path("/kaggle/working").rglob("common/models.py"))
cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
assert cands, "Attach the parametric-fixes zip (common/)"

def _ok(p: Path) -> bool:
    ju = (p.parent / "juice.py").read_text(encoding="utf-8", errors="ignore")
    ja = (p.parent / "juice_attn.py").read_text(encoding="utf-8", errors="ignore")
    return "dense_linear_weight_T" in ja and "identified_from_enlarge" in ju

cands.sort(key=_ok, reverse=True)
assert _ok(cands[0]), "Need updated common/ from kaggle_juice_stage2_int4_parametric_fixes.zip"
REPO = cands[0].parent.parent
dst = Path("/kaggle/working/common")
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(REPO / "common", dst)
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")
print("REPO", REPO)

try:
    from huggingface_hub import login
    from kaggle_secrets import UserSecretsClient
    login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
except Exception as e:
    print("HF login skipped", type(e).__name__, e)
"""
)

md("## 2 — Locate heads, split, INT4 scores, checkpoint")
code(
    r"""OUT_DIR = Path("/kaggle/working/juice_stage2_int4_parametric_fixes")
OUT_DIR.mkdir(parents=True, exist_ok=True)
INT4_MODEL_ID_OVERRIDE = None

def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if "native_id" in str(p) or "diagnostics" in str(p) or "directional" in str(p) else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

CLEAN_PATH = find_one("heads_dir_parametric_k2_clean.json")
CTX10_PATH = find_one("heads_dir_context_k10.json")
SPLIT_PATH = find_one("split.json")
DIAG_PATH = find_one("id_diagnostics_raw.json")
DATA_NAME = "conflictqa_pilot250_popular.jsonl"
data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
assert data_hits
DATA_PATH = data_hits[0]

from common.conflictqa import load_conflict_items
from common.juice import load_identified_heads, head_keys
from common.models import resolve_gptq_8b_base_model_id

INT4_MODEL_ID = resolve_gptq_8b_base_model_id(INT4_MODEL_ID_OVERRIDE)
print("INT4_MODEL_ID", INT4_MODEL_ID)

split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
eval_ids = set(split["eval_ids"])
eval_items = [it for it in load_conflict_items(DATA_PATH) if it.id in eval_ids]
assert len(eval_items) == split["n_eval"]
clean = load_identified_heads(CLEAN_PATH)
ctx10 = load_identified_heads(CTX10_PATH)
print("clean K2", head_keys(clean.enlarge))
print("ctx K10", head_keys(ctx10.enlarge))
print("n_eval", len(eval_items))
"""
)

md("## 3 — Check 2: directional_par ranks 1–20 (no GPU)")
code(
    r"""import numpy as np

raw = json.loads(DIAG_PATH.read_text(encoding="utf-8"))
ctx_s, par_s = raw["scores_pos_context"], raw["scores_pos_parametric"]
keys = set(ctx_s) | set(par_s)
rows = []
for k in keys:
    c = float(ctx_s.get(k, 0.0))
    p = float(par_s.get(k, 0.0))
    rows.append({"head": k, "ctx_pos": c, "par_pos": p, "dir_par": p - c})
rows.sort(key=lambda r: -r["dir_par"])
noise = float(np.median([max(abs(ctx_s.get(k, 0.0)), abs(par_s.get(k, 0.0))) for k in keys]))
thr5 = 5 * noise
print("noise proxy median max(|ctx|,|par|)=%.4e  5x=%.4e" % (noise, thr5))
print("%3s %-10s %10s %12s %12s flags" % ("rk", "head", "dir_par", "par_pos", "ctx_pos"))
top20 = []
for i, r in enumerate(rows[:20], 1):
    flags = []
    if r["par_pos"] > 0:
        flags.append("par_pos>0")
    if r["par_pos"] > 0 and r["dir_par"] > 0:
        flags.append("CLEAN")
    if r["par_pos"] >= thr5:
        flags.append("par>=5x_noise")
    print("%3d %-10s %+10.4f %+12.4e %+12.4e %s" % (
        i, r["head"], r["dir_par"], r["par_pos"], r["ctx_pos"], " ".join(flags)
    ))
    top20.append({**r, "rank": i, "flags": flags})

pos = [r for r in rows if r["par_pos"] > 0]
print("\npar_pos>0: %d / %d heads" % (len(pos), len(rows)))
print("=== par_pos>0 ranked by dir_par (top 15) ===")
flagged = []
for i, r in enumerate(pos[:15], 1):
    rk = next(j for j, x in enumerate(rows, 1) if x["head"] == r["head"])
    mag = "MEANINGFUL" if r["par_pos"] >= thr5 else "near-noise"
    print("%3d %-10s dir_rank=%3d par=%+.4e ctx=%+.4e %s" % (
        i, r["head"], rk, r["par_pos"], r["ctx_pos"], mag
    ))
    if r["par_pos"] >= thr5:
        flagged.append({**r, "dir_rank": rk, "magnitude": mag})

print("\nFLAGGED: positive par_pos with magnitude >= 5x noise, in/near top-20 dir_par")
print(json.dumps(flagged, indent=2))

scan = {
    "noise_proxy_median": noise,
    "par_pos_meaningful_threshold_5x": thr5,
    "directional_par_top20": top20,
    "flagged_positive_par_pos": flagged,
    "n_par_pos": len(pos),
}
(OUT_DIR / "directional_par_top20.json").write_text(json.dumps(scan, indent=2), encoding="utf-8")
print("wrote", OUT_DIR / "directional_par_top20.json")
"""
)

md("## 4 — Load INT4 (GPU)")
code(
    r"""from common.models import load_model_and_tokenizer
from common.juice import wrap_llama_attention_for_juice

print("Loading INT4", INT4_MODEL_ID)
model, tokenizer = load_model_and_tokenizer(
    "int4",
    model_id_int4=INT4_MODEL_ID,
    device_map={"": 0},
)
wrap_llama_attention_for_juice(model)
print("wrapped")
"""
)

md("## 5 — Checks 4 + 1 + 3: L1_H4, then alpha grid, then suppress ctx K10")
code(
    r"""from common.inference import greedy_generate
from common.juice import (
    HeadRef,
    IdentifiedHeads,
    apply_juice_steering,
    conflict_eval_prompt,
    context_follow_rate,
    identified_from_enlarge,
)
from common.scoring import label_conflict_response
from common.stats_analysis import bootstrap_paired_follow_delta
from tqdm import tqdm

MAX_NEW_TOKENS = 16
PROMPT_FORMAT = "juice_coherent"
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0]
TRANSFER_ALPHA1_PP = -2.0  # prior INT4 transfer, n=50

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

baseline = []
for item in tqdm(eval_items, desc="baseline"):
    prompt = conflict_eval_prompt(tokenizer, item, prompt_format=PROMPT_FORMAT)
    text = greedy_generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
    lab = label_conflict_response(text, item.parametric_answer, item.context_answer)
    baseline.append({"id": item.id, "text": text, "label": lab, "follows_context": lab == "context"})
r_base = context_follow_rate(baseline)
c_base = coherence_rate(baseline)
print(f"baseline fc={r_base:.3f} coh={c_base:.3f} {label_counts(baseline)}")
(OUT_DIR / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

summary = {
    "cell": "int4_parametric_recovery",
    "n_eval": len(eval_items),
    "baseline": {"follows_context": r_base, "coherence": c_base, "label_counts": label_counts(baseline)},
    "alpha_grid": {},
    "l1_h4": {},
    "suppress_context_k10": {},
    "sign_flip_read": None,
}

print("\n=== CHECK 4: L1_H4 alone (K=1, alpha_plus=1) ===")
print("FP16: raw par rank 1  par_pos=+0.0577  ctx_pos=-0.0392  dir_par=+0.0969  dir_rank=29")
print("INT4: raw par rank 1  par_pos=+0.0578  ctx_pos=-0.0581  dir_par=+0.1159  dir_rank=14")
print("same-head (raw peak survived GPTQ); unused in Cell-3 because directional rerank preferred L20/L31")
ident_l1 = identified_from_enlarge(
    [HeadRef(1, 4)], precision="int4_l1h4", K=1, steer_target="parametric"
)
rows = apply_juice_steering(
    model, tokenizer, eval_items, ident_l1,
    alpha_plus=1.0, alpha_minus=0.0,
    max_new_tokens=MAX_NEW_TOKENS, prompt_format=PROMPT_FORMAT,
)
r = context_follow_rate(rows)
coh = coherence_rate(rows)
d_pp = (r - r_base) * 100
boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=1000, seed=42)
summary["l1_h4"] = {
    "heads": ["L1_H4"],
    "alpha_plus": 1.0,
    "follows_context": r,
    "delta_pp_vs_baseline": d_pp,
    "coherence": coh,
    "label_counts": label_counts(rows),
    "bootstrap": boot["delta"],
    "fp16_id": {
        "raw_par_rank": 1,
        "dir_par_rank": 29,
        "par_pos": 0.05771769270844137,
        "ctx_pos": -0.039223989231407096,
        "dir_par": 0.09694168193984846,
        "in_fp16_raw_enlarge": True,
        "in_fp16_directional_cell3": False,
    },
    "int4_id": {
        "raw_par_rank": 1,
        "dir_par_rank": 14,
        "par_pos": 0.0578033780782587,
        "ctx_pos": -0.058143947950766356,
        "dir_par": 0.11594732602902505,
    },
    "interpretation": (
        "same-head: FP16 and INT4 raw parametric peak is L1_H4 with matching par_pos. "
        "Directional Cell-3 skipped it. If INT4 delta is negative and excl. 0, parametric "
        "steering is recoverable via the raw-ID head, not the directional clean pair."
    ),
}
(OUT_DIR / "steer_l1_h4.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
ci = boot["delta"]["ci95_pp"]
print(
    f"L1_H4  fc={r:.3f} d={d_pp:+.1f} CI=[{ci[0]:.1f},{ci[1]:.1f}] "
    f"coh={coh:.3f} excl0={boot['delta']['excludes_zero']}"
)
print("  (parametric success = negative delta)")

print("\n=== CHECK 1: alpha grid on L20_H14, L26_H15 ===")
print(f"{'alpha':>8s} {'fc':>8s} {'d_pp':>8s} {'CI':>18s} {'coh':>6s} {'excl0':>6s}")
prev_d = None
monotonic_wrong = []  # delta becomes more POSITIVE (more context) as alpha rises
for a in ALPHAS:
    rows = apply_juice_steering(
        model, tokenizer, eval_items, clean,
        alpha_plus=a, alpha_minus=0.0,
        max_new_tokens=MAX_NEW_TOKENS, prompt_format=PROMPT_FORMAT,
    )
    r = context_follow_rate(rows)
    coh = coherence_rate(rows)
    d_pp = (r - r_base) * 100
    boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=1000, seed=42)
    entry = {
        "alpha_plus": a,
        "heads": [h.key() for h in clean.enlarge],
        "follows_context": r,
        "delta_pp_vs_baseline": d_pp,
        "coherence": coh,
        "label_counts": label_counts(rows),
        "bootstrap": boot["delta"],
    }
    summary["alpha_grid"][f"alpha_{a}"] = entry
    (OUT_DIR / f"steer_clean_alpha_{a}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ci = boot["delta"]["ci95_pp"]
    print(f"{a:8.1f} {r:8.3f} {d_pp:8.1f} [{ci[0]:6.1f},{ci[1]:6.1f}] {coh:6.3f} {str(boot['delta']['excludes_zero']):>6s}")
    if prev_d is not None and d_pp > prev_d + 0.5:
        monotonic_wrong.append({"from_alpha": None, "d_from": prev_d, "to_alpha": a, "d_to": d_pp})
        monotonic_wrong[-1]["from_alpha"] = ALPHAS[ALPHAS.index(a) - 1]
    prev_d = d_pp

grid_ds = [summary["alpha_grid"][f"alpha_{a}"]["delta_pp_vs_baseline"] for a in ALPHAS]
# Wrong direction = more context-following as we enlarge parametric heads (delta rising toward +)
wrong = (grid_ds[-1] > grid_ds[0] + 1) or (max(grid_ds) > 0 and grid_ds[-1] >= max(0.0, grid_ds[0]))
summary["sign_flip_read"] = {
    "deltas_pp_in_alpha_order": grid_ds,
    "alphas": ALPHAS,
    "effect_moves_wrong_way_as_alpha_increases": bool(
        grid_ds[-1] > grid_ds[0] + 1.0
    ),
    "any_positive_delta": any(d > 0 for d in grid_ds),
    "coherence_collapse": any(summary["alpha_grid"][f"alpha_{a}"]["coherence"] < 0.90 for a in ALPHAS),
    "note": (
        "Parametric enlarge should LOWER follows_context (negative delta). "
        "If delta rises (toward +) as alpha increases, INT4 sign-flip is real and rescale will not recover FP16."
    ),
}
print("\nsign_flip_read", json.dumps(summary["sign_flip_read"], indent=2))

print("\n=== CHECK 3: suppress context K10 (H−, alpha_minus=1) ===")
ident_sup = IdentifiedHeads(
    enlarge=[],
    suppress=list(ctx10.enlarge),
    scores_pos={},
    scores_neg={},
    precision="int4_suppress_ctx_k10",
    K=len(ctx10.enlarge),
    steer_target="parametric",
    n_calibration=0,
    pos_alphas=[],
    neg_alphas=[],
)
rows = apply_juice_steering(
    model, tokenizer, eval_items, ident_sup,
    alpha_plus=0.0, alpha_minus=1.0,
    max_new_tokens=MAX_NEW_TOKENS, prompt_format=PROMPT_FORMAT,
)
r = context_follow_rate(rows)
coh = coherence_rate(rows)
d_pp = (r - r_base) * 100
boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=1000, seed=42)
summary["suppress_context_k10"] = {
    "heads": [h.key() for h in ctx10.enlarge],
    "mechanism": "H- on FP16 directional context K10, alpha_minus=1, enlarge empty",
    "follows_context": r,
    "delta_pp_vs_baseline": d_pp,
    "coherence": coh,
    "label_counts": label_counts(rows),
    "bootstrap": boot["delta"],
}
(OUT_DIR / "steer_suppress_ctx_k10.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
ci = boot["delta"]["ci95_pp"]
print(f"suppress ctx K10  fc={r:.3f} d={d_pp:+.1f} CI=[{ci[0]:.1f},{ci[1]:.1f}] coh={coh:.3f} excl0={boot['delta']['excludes_zero']}")
print("  (parametric success would be negative delta / less context-following)")

(OUT_DIR / "int4_parametric_fixes_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\n=== SUMMARY TABLE ===")
print(f"{'condition':28s} {'d_pp':>8s} {'CI lo':>8s} {'CI hi':>8s} {'coh':>6s}")
print(f"{'baseline':28s} {0.0:8.1f} {'':>8s} {'':>8s} {c_base:6.3f}")
e = summary["l1_h4"]
b = e["bootstrap"]
print(f"{'L1_H4 K=1 a=1':28s} {e['delta_pp_vs_baseline']:8.1f} {b['ci95_pp'][0]:8.1f} {b['ci95_pp'][1]:8.1f} {e['coherence']:6.3f}")
for a in ALPHAS:
    e = summary["alpha_grid"][f"alpha_{a}"]
    b = e["bootstrap"]
    print(f"{'clean K2 a='+str(a):28s} {e['delta_pp_vs_baseline']:8.1f} {b['ci95_pp'][0]:8.1f} {b['ci95_pp'][1]:8.1f} {e['coherence']:6.3f}")
e = summary["suppress_context_k10"]
b = e["bootstrap"]
print(f"{'suppress ctx K10':28s} {e['delta_pp_vs_baseline']:8.1f} {b['ci95_pp'][0]:8.1f} {b['ci95_pp'][1]:8.1f} {e['coherence']:6.3f}")
print("done", OUT_DIR)
"""
)

md("## 6 — Unload")
code(
    r"""from common.models import unload_model
unload_model(model)
print("unloaded")
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
out = Path(__file__).resolve().parent / "kaggle_stage2_int4_parametric_fixes.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
