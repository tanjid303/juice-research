"""Generate juice_repro/kaggle_stage2_int4_native_id.ipynb

Native INT4 head ID + random-K control + Jaccard vs FP16 (with |D|=4 bootstrap).
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
    """# Stage 2 — Native INT4 ID + random-K + Jaccard

Identify JuICE heads **directly on INT4** (same protocol as FP16 base), then:

1. **Native ID** — |D|=4, 3-prompt-type JuICE prompts. Do L20_H14 / L26_H15 and the FP16 context set come back?
2. **Random-K** — same α=1, same K, randomly selected heads (not identified). Is transfer / context robustness head-specific?
3. **Jaccard** — FP16 vs INT4 native sets vs a **|D|=4 calibration bootstrap** noise floor (20+ draws from saved per-item scores; no extra forwards).

**Model:** self-quantized GPTQ-INT4 Llama-3.1-8B base  
**Calib IDs:** first 4 of `split.json` (same as FP16)  
**Eval:** same 50 held-out items, `juice_coherent`, α+=1, α−=0

Transfer reference (n=50, 1000-draw paired bootstrap):

| Condition | INT4 Δ | 95% CI | excl. 0? | FP16 Δ | FP16 CI |
|-----------|---------|--------|----------|--------|---------|
| context K10 | +12 | [+4, +22] | yes | +10 | [+2, +20] |
| context K5 | +6 | [−2, +16] | no | +6 | [0, +12] |
| parametric clean K2 | **−2** | **[−6, 0]** | **no** | −8 | [−16, −2] |
| parametric K3 | −4 | [−12, +6] | no | −6 | [−18, +6] |

## Before Run All

1. GPU **T4 ×2**, Internet ON
2. Attach this zip **and** the quantize notebook output (`llama31_8b_base_gptq_int4`)
3. Upload `juice_repro/kaggle_stage2_int4_native_id.ipynb`
4. Save & Run All (~3–6 h: ID is the long part)
"""
)

md("## 1 — Setup")
code(
    r"""import os, sys, subprocess, json, re, random
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
assert cands, "Attach the native-ID dataset (must contain common/)"

def _juice_attn_ok(models_py: Path) -> bool:
    ja = models_py.parent / "juice_attn.py"
    if not ja.is_file():
        return False
    txt = ja.read_text(encoding="utf-8", errors="ignore")
    ju = models_py.parent / "juice.py"
    ju_txt = ju.read_text(encoding="utf-8", errors="ignore") if ju.is_file() else ""
    return "dense_linear_weight_T" in txt and "per_item_pos_scores" in ju_txt

cands.sort(key=lambda p: (_juice_attn_ok(p), "native" in str(p).lower()), reverse=True)
assert _juice_attn_ok(cands[0]), (
    "Need common/juice.py with per-item ID scores + GPTQ o_proj dequant. "
    "Attach the NEW kaggle_juice_stage2_int4_native_id.zip."
)
REPO = cands[0].parent.parent
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
from common.juice import bootstrap_id_jaccard  # noqa: F401
print("juice_attn GPTQ dequant + per-item ID: OK")

try:
    from huggingface_hub import login
    from kaggle_secrets import UserSecretsClient
    login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
    print("HF login ok")
except Exception as e:
    print("HF login skipped:", type(e).__name__, e)
"""
)

md("## 2 — Locate split, data, FP16 heads, INT4 checkpoint")
code(
    r"""OUT_DIR = Path("/kaggle/working/juice_stage2_int4_native_id")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "directional_heads").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "id_checkpoints").mkdir(parents=True, exist_ok=True)

INT4_MODEL_ID_OVERRIDE = None

print("=== /kaggle/input (top-level) ===")
inp = Path("/kaggle/input")
if inp.exists():
    for p in sorted(inp.iterdir()):
        print(" ", "DIR " if p.is_dir() else "FILE", p)

def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if "directional" in str(p) or "base_id" in str(p) else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

HEADS_FP16 = {
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
from common.juice import load_identified_heads, head_keys
from common.models import resolve_gptq_8b_base_model_id

INT4_MODEL_ID = resolve_gptq_8b_base_model_id(INT4_MODEL_ID_OVERRIDE)
print("INT4_MODEL_ID", INT4_MODEL_ID)

split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
calib_ids = list(split["calib_ids"])[:4]
eval_ids = set(split["eval_ids"])
all_items = load_conflict_items(DATA_PATH)
by_id = {it.id: it for it in all_items}
calib_items = [by_id[i] for i in calib_ids if i in by_id]
eval_items = [it for it in all_items if it.id in eval_ids]
assert len(calib_items) == 4, [it.id for it in calib_items]
assert len(eval_items) == split["n_eval"]
print("calib_ids", calib_ids)
print("n_eval", len(eval_items))

fp16_sets = {}
for tag, path in HEADS_FP16.items():
    ident = load_identified_heads(path)
    keys = head_keys(ident.enlarge)
    print("FP16", tag, "->", keys)
    fp16_sets[tag] = keys

FP16_JACCARD_SETS = {
    "dir_context_k10": fp16_sets["ctx_k10"],
    "dir_context_k5": fp16_sets["ctx_k5"],
    "dir_parametric_k2": fp16_sets["par_k2"],
    "dir_parametric_k3": fp16_sets["par_k3"],
    "dir_parametric_k2_clean": fp16_sets["par_k2_clean"],
}
"""
)

md("## 3 — Load INT4 base (GPTQ, GPU 0)")
code(
    r"""from common.models import load_model_and_tokenizer
from common.juice import wrap_llama_attention_for_juice

print("Loading INT4", INT4_MODEL_ID)
model, tokenizer = load_model_and_tokenizer(
    "int4",
    model_id_int4=INT4_MODEL_ID,
    device_map={"": 0},
)
print("device_map", getattr(model, "hf_device_map", None))
print("config.quantization_config", getattr(model.config, "quantization_config", None))
wrap_llama_attention_for_juice(model)
print("wrapped JuICE attention (GPTQ o_proj dequant cached)")
"""
)

md("## 4 — Native INT4 ID (|D|=4, 3 prompt types)")
code(
    r"""from common.juice import (
    FAST_NEG_ALPHAS,
    FAST_POS_ALPHAS,
    identify_both_directions_juice_prompts,
    print_identified,
    save_identified_heads,
    directional_rerank_report,
    directional_top_keys,
    directional_par_clean_keys,
    rank_of_key,
    jaccard,
)
from common.models import unload_model
import numpy as np

K = 10
CANONICAL_PAR = ["L20_H14", "L26_H15"]
FP16_PAR_NOISY = ["L20_H14", "L31_H14"]

heads_ctx, heads_par, diag = identify_both_directions_juice_prompts(
    model,
    tokenizer,
    calib_items,
    precision="int4_base",
    K=K,
    pos_alphas=FAST_POS_ALPHAS,
    neg_alphas=FAST_NEG_ALPHAS,
    seed=42,
    checkpoint_dir=OUT_DIR / "id_checkpoints",
    require_all_prompt_types_positive=True,
)
save_identified_heads(OUT_DIR / "heads_context_int4.json", heads_ctx)
save_identified_heads(OUT_DIR / "heads_parametric_int4.json", heads_par)

# Mean score maps (compact). Per-item maps are large — save separately.
raw_diag = {
    "scores_pos_context": diag["scores_pos_context"],
    "scores_pos_parametric": diag["scores_pos_parametric"],
    "scores_neg_context": diag["scores_neg_context"],
    "scores_neg_parametric": diag["scores_neg_parametric"],
    "n_calibration": diag["n_calibration"],
    "prompt_types": diag["prompt_types"],
    "require_all_prompt_types_positive": diag["require_all_prompt_types_positive"],
}
(OUT_DIR / "id_diagnostics_raw.json").write_text(json.dumps(raw_diag, indent=2), encoding="utf-8")
(OUT_DIR / "id_per_item_scores.json").write_text(
    json.dumps(diag["per_item_pos_scores"]),
    encoding="utf-8",
)
print("saved per-item scores for", len(diag["per_item_pos_scores"]), "calib items")
print_identified(heads_ctx)
print_identified(heads_par)
"""
)

md("## 5 — Directional re-rank + FP16 head recovery")
code(
    r"""drep = directional_rerank_report(heads_ctx.scores_pos, heads_par.scores_pos, K=10)
(OUT_DIR / "directional_rerank_int4.json").write_text(json.dumps(drep, indent=2), encoding="utf-8")

ctx10 = [r["head"] for r in drep["directional_ctx_top"]]
ctx5 = ctx10[:5]
par_all = [r["head"] for r in drep["directional_par_top"]]
par2 = par_all[:2]
par3 = par_all[:3]
par2_clean = directional_par_clean_keys(heads_ctx.scores_pos, heads_par.scores_pos, K=2)

int4_sets = {
    "dir_context_k10": ctx10,
    "dir_context_k5": ctx5,
    "dir_parametric_k2": par2,
    "dir_parametric_k3": par3,
    "dir_parametric_k2_clean": par2_clean,
}

print("=== RAW enlarge (INT4) ===")
print("ctx", [h.key() for h in heads_ctx.enlarge])
print("par", [h.key() for h in heads_par.enlarge])
print("raw enlarge Jaccard ctx-vs-par", jaccard(heads_ctx.enlarge, heads_par.enlarge))

print("\n=== DIRECTIONAL (INT4) ===")
print("ctx K10", ctx10)
print("ctx K5 ", ctx5)
print("par K2 ", par2)
print("par K3 ", par3)
print("par clean K2 (par_pos>0 & dir_par>0)", par2_clean)
print("directional top-10 Jaccard ctx-vs-par", drep["jaccard"])

print("\n=== Do FP16 heads come back? ===")
print("FP16 context K10", FP16_JACCARD_SETS["dir_context_k10"])
print("  overlap K10", sorted(set(ctx10) & set(FP16_JACCARD_SETS["dir_context_k10"])))
print("  Jaccard K10", len(set(ctx10) & set(FP16_JACCARD_SETS["dir_context_k10"])) / len(set(ctx10) | set(FP16_JACCARD_SETS["dir_context_k10"])))
print("FP16 context K5", FP16_JACCARD_SETS["dir_context_k5"])
print("  overlap K5", sorted(set(ctx5) & set(FP16_JACCARD_SETS["dir_context_k5"])))

print("\nCanonical parametric L20_H14 / L26_H15 ranks on INT4 directional_par:")
for k in CANONICAL_PAR + ["L31_H14"]:
    rnk = rank_of_key(heads_ctx.scores_pos, heads_par.scores_pos, k, "parametric")
    c = heads_ctx.scores_pos.get(k, float("nan"))
    p = heads_par.scores_pos.get(k, float("nan"))
    print(f"  {k:10s} rank={rnk}  ctx_pos={c:+.4e} par_pos={p:+.4e} dir_par={p-c:+.4e}")

print("\nRaw-ID enlarge: are canonical heads in INT4 parametric enlarge?")
par_enc = {h.key() for h in heads_par.enlarge}
ctx_enc = {h.key() for h in heads_ctx.enlarge}
for k in CANONICAL_PAR + FP16_JACCARD_SETS["dir_context_k10"]:
    print(f"  {k:10s} in_par_enlarge={k in par_enc}  in_ctx_enlarge={k in ctx_enc}")

def write_dir_heads(keys, steer_target, K, name, scores):
    enlarge = []
    for k in keys[:K]:
        layer, head = k.replace("L", "").split("_H")
        enlarge.append({"layer": int(layer), "head": int(head)})
    payload = {
        "enlarge": enlarge,
        "suppress": [],
        "scores_pos": scores,
        "scores_neg": {},
        "precision": "int4_base",
        "K": K,
        "steer_target": steer_target,
        "n_calibration": 4,
        "pos_alphas": list(FAST_POS_ALPHAS),
        "neg_alphas": list(FAST_NEG_ALPHAS),
        "selection_method": f"directional_rerank_int4_{name}",
    }
    path = OUT_DIR / "directional_heads" / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", path, "->", keys[:K])

write_dir_heads(ctx10, "context", 10, "heads_dir_context_k10.json", heads_ctx.scores_pos)
write_dir_heads(ctx5, "context", 5, "heads_dir_context_k5.json", heads_ctx.scores_pos)
write_dir_heads(par2, "parametric", 2, "heads_dir_parametric_k2.json", heads_par.scores_pos)
write_dir_heads(par3, "parametric", 3, "heads_dir_parametric_k3.json", heads_par.scores_pos)
write_dir_heads(par2_clean, "parametric", 2, "heads_dir_parametric_k2_clean.json", heads_par.scores_pos)
"""
)

md("## 6 — Jaccard vs FP16 + |D|=4 bootstrap noise floor")
code(
    r"""from common.juice import bootstrap_id_jaccard

jac = bootstrap_id_jaccard(
    diag["per_item_pos_scores"],
    fp16_sets=FP16_JACCARD_SETS,
    int4_sets=int4_sets,
    n_boot=25,
    seed=42,
)
# Drop bulky value lists from the printed table copy; keep them on disk.
(OUT_DIR / "jaccard_bootstrap.json").write_text(json.dumps(jac, indent=2), encoding="utf-8")

print("n_boot", jac["n_boot"], "calib", jac["item_ids"])
print(f"{'spec':28s} {'obs J':>8s} {'noise mean':>12s} {'noise CI':>18s} {'vs INT4 ID':>12s}")
for spec, row in jac["by_spec"].items():
    noise = row["bootstrap_int4_vs_int4"]
    vs_id = row["bootstrap_vs_int4_id"]
    print(
        f"{spec:28s} {row['observed_fp16_vs_int4']:8.3f} "
        f"{noise['mean']:12.3f} "
        f"[{noise['ci95'][0]:.3f}, {noise['ci95'][1]:.3f}] "
        f"{vs_id['mean']:12.3f}"
    )
    print("  FP16", row["fp16_set"])
    print("  INT4", row["int4_set"])

# Same-set noise floor interpretation:
# if observed FP16-INT4 Jaccard is within bootstrap_int4_vs_int4 CI,
# the head-set difference is within |D|=4 identification variance.
print("\n=== noise-floor read ===")
for spec, row in jac["by_spec"].items():
    obs = row["observed_fp16_vs_int4"]
    lo, hi = row["bootstrap_int4_vs_int4"]["ci95"]
    inside = lo <= obs <= hi
    print(
        f"{spec}: observed J={obs:.3f} vs INT4-vs-INT4 noise [{lo:.3f}, {hi:.3f}] "
        f"{'WITHIN noise' if inside else 'BELOW/OUTSIDE noise floor'}"
    )
"""
)

md("## 7 — Random-K control (same alpha, same K, random heads)")
code(
    r"""from common.inference import greedy_generate
from common.juice import (
    apply_juice_steering,
    context_follow_rate,
    identified_from_enlarge,
    random_head_control,
    model_dims,
    HeadRef,
    conflict_eval_prompt,
)
from common.scoring import label_conflict_response
from common.stats_analysis import bootstrap_paired_follow_delta
from tqdm import tqdm

MAX_NEW_TOKENS = 16
ALPHA_PLUS = 1.0
ALPHA_MINUS = 0.0
PROMPT_FORMAT = "juice_coherent"
N_SHOW = 5

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

def key_to_ref(k: str) -> HeadRef:
    layer, head = k.replace("L", "").split("_H")
    return HeadRef(int(layer), int(head))

baseline = []
for item in tqdm(eval_items, desc="baseline"):
    prompt = conflict_eval_prompt(tokenizer, item, prompt_format=PROMPT_FORMAT)
    text = greedy_generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
    lab = label_conflict_response(text, item.parametric_answer, item.context_answer)
    baseline.append({"id": item.id, "text": text, "label": lab, "follows_context": lab == "context"})

r_base = context_follow_rate(baseline)
c_base = coherence_rate(baseline)
print(f"INT4 baseline fc={r_base:.3f} coherence={c_base:.3f} counts={label_counts(baseline)}")
show_examples(baseline, "INT4 BASELINE")
(OUT_DIR / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

n_layers, n_heads, _ = model_dims(model)
print("dims", n_layers, n_heads)

# Exclude the FP16 identified heads of that direction so random is not accidentally the transfer set.
exclude_ctx = [key_to_ref(k) for k in FP16_JACCARD_SETS["dir_context_k10"]]
exclude_par = [key_to_ref(k) for k in sorted(set(
    FP16_JACCARD_SETS["dir_parametric_k2"]
    + FP16_JACCARD_SETS["dir_parametric_k3"]
    + FP16_JACCARD_SETS["dir_parametric_k2_clean"]
))]

RANDOM_SPECS = [
    ("random_context_k10", "context", 10, 42, exclude_ctx, 12.0),
    ("random_context_k5", "context", 5, 43, exclude_ctx, 6.0),
    ("random_parametric_k2", "parametric", 2, 44, exclude_par, -2.0),
    ("random_parametric_k3", "parametric", 3, 45, exclude_par, -4.0),
]

random_summary = {
    "cell": "int4_random_k_control",
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
}

for tag, target, Krand, seed, exclude, identified_delta_pp in RANDOM_SPECS:
    heads = random_head_control(n_layers, n_heads, Krand, seed=seed, exclude=exclude)
    ident = identified_from_enlarge(
        heads, precision="int4_random", K=Krand, steer_target=target
    )
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
    boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=1000, seed=42)
    entry = {
        "heads": [h.key() for h in ident.enlarge],
        "steer_target": target,
        "K": Krand,
        "seed": seed,
        "follows_context": r,
        "delta_pp_vs_baseline": d_pp,
        "coherence": coh,
        "label_counts": label_counts(rows),
        "identified_transfer_delta_pp": identified_delta_pp,
        "bootstrap": boot["delta"],
    }
    random_summary["conditions"][tag] = entry
    print(json.dumps({k: v for k, v in entry.items() if k != "bootstrap"}, indent=2))
    print("bootstrap delta_pp", boot["delta"]["point_pp"], "CI", boot["delta"]["ci95_pp"], "excl0", boot["delta"]["excludes_zero"])
    show_examples(rows, tag)
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

(OUT_DIR / "int4_random_k_summary.json").write_text(json.dumps(random_summary, indent=2), encoding="utf-8")
print("\n=== RANDOM-K vs identified transfer ===")
print(f"{'condition':24s} {'rand d':>8s} {'CI lo':>8s} {'CI hi':>8s} {'id d':>8s} {'coh':>6s}")
for tag, entry in random_summary["conditions"].items():
    b = entry["bootstrap"]
    print(
        f"{tag:24s} {entry['delta_pp_vs_baseline']:8.1f} "
        f"{b['ci95_pp'][0]:8.1f} {b['ci95_pp'][1]:8.1f} "
        f"{entry['identified_transfer_delta_pp']:8.1f} {entry['coherence']:6.3f}"
    )
"""
)

md("## 8 — Combined summary")
code(
    r"""id_compare = {
    "cell": "int4_native_id",
    "identify_on": "int4_base",
    "model": INT4_MODEL_ID,
    "n_calib": 4,
    "calib_ids": calib_ids,
    "raw": {
        "ctx_enlarge": [h.key() for h in heads_ctx.enlarge],
        "par_enlarge": [h.key() for h in heads_par.enlarge],
        "enlarge_jaccard_ctx_vs_par": jaccard(heads_ctx.enlarge, heads_par.enlarge),
        "peak_score_pos_context": max(heads_ctx.scores_pos.values()) if heads_ctx.scores_pos else 0.0,
        "peak_score_pos_parametric": max(heads_par.scores_pos.values()) if heads_par.scores_pos else 0.0,
    },
    "directional": int4_sets,
    "fp16_sets": FP16_JACCARD_SETS,
    "canonical_parametric_ranks": {
        k: rank_of_key(heads_ctx.scores_pos, heads_par.scores_pos, k, "parametric")
        for k in CANONICAL_PAR + ["L31_H14"]
    },
    "canonical_in_raw_par_enlarge": {
        k: k in {h.key() for h in heads_par.enlarge} for k in CANONICAL_PAR
    },
    "jaccard_bootstrap_file": "jaccard_bootstrap.json",
}

(OUT_DIR / "int4_native_id_summary.json").write_text(json.dumps(id_compare, indent=2), encoding="utf-8")

print("=== NATIVE ID TAKEAWAY ===")
print(json.dumps(id_compare, indent=2)[:4000])
print("\ndone", OUT_DIR)
unload_model(model)
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
out = Path(__file__).resolve().parent / "kaggle_stage2_int4_native_id.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
