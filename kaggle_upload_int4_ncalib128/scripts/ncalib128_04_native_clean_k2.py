"""Steer the native INT4 clean-filter pair L16_H30 + L17_H27 on the N_CALIB=128 checkpoint.

This pair came out of the native INT4 directional clean filter (par_pos>0 AND dir_par>0)
and has never been applied — flag 2 in writeup/stage1_stage2_consolidated.md.

Spec (identical to the other N=128 transfer cells except the head set):
  heads   K=2, L16_H30 + L17_H27, steer_target=parametric
  gain    alpha_plus=1.0, alpha_minus=0.0
  eval    same 50 held-out ids (split.json, seed 42)
  prompt  juice_coherent, greedy, max_new_tokens=16
  stats   bootstrap_paired_follow_delta, n_boot=1000, seed=42

Outputs (same schema as the other transfer cells):
  int4_native_clean_k2_summary.json
  int4_native_clean_k2_bootstrap.json
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGE_T0 = time.time()

TAG = "steer_int4_native_parametric_k2_clean"
EXPECTED_HEADS = ["L16_H30", "L17_H27"]


def elapsed(stage: str) -> None:
    print(f"[elapsed] {stage}: {(time.time() - STAGE_T0) / 60:.1f} min since start")


def _bootstrap_repo() -> Path:
    cands = list(Path("/kaggle/working").rglob("common/models.py"))
    cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
    assert cands, "Attach the ncalib128 dataset (must contain common/)"

    def _juice_attn_ok(models_py: Path) -> bool:
        ja = models_py.parent / "juice_attn.py"
        if not ja.is_file():
            return False
        return "dense_linear_weight_T" in ja.read_text(encoding="utf-8", errors="ignore")

    cands.sort(key=lambda p: (_juice_attn_ok(p), "ncalib" in str(p).lower()), reverse=True)
    assert _juice_attn_ok(cands[0]), (
        "common/juice_attn.py is missing dense_linear_weight_T (GPTQ o_proj dequant). "
        "Attach kaggle_juice_stage2_int4_ncalib128.zip."
    )
    repo = cands[0].parent.parent
    dst_common = Path("/kaggle/working/common")
    if dst_common.exists():
        shutil.rmtree(dst_common)
    shutil.copytree(repo / "common", dst_common)
    if "/kaggle/working" not in sys.path:
        sys.path.insert(0, "/kaggle/working")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    print("REPO", repo)
    from common.juice_attn import dense_linear_weight_T  # noqa: F401

    print("juice_attn GPTQ o_proj dequant: OK")
    return repo


def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if "directional" in str(p) or "base_id" in str(p) else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]


def resolve_ncalib128_checkpoint(explicit: str | None = None) -> str:
    """Only accept the N_CALIB=128 dir. Never fall back to the N64 path."""
    from common.models import _looks_like_gptq_dir

    named = "llama31_8b_base_gptq_int4_ncalib128"
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if not root.exists():
            continue
        candidates.append(root / named)
        try:
            for d in root.rglob(named):
                if d.is_dir():
                    candidates.append(d)
        except OSError:
            pass
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = str(path.resolve()) if path.exists() else str(path)
        except OSError:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_gptq_dir(path):
            print("Resolved N_CALIB=128 GPTQ ->", path)
            return str(path)
    raise FileNotFoundError(
        "N_CALIB=128 checkpoint not found. Expected a folder named "
        f"{named} with config.json + *.safetensors.\n"
        "Run script 01 first, or attach that notebook output.\n"
        f"Tried {list(seen)[:12]}"
    )


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


def show_examples(rows, title, n_show=5):
    print(f"\n===== {title}: {n_show} raw examples =====")
    for r in rows[:n_show]:
        print(f"- {r['id']} [{r['label']}] {r['text']!r}")


def _load_reference_baseline() -> list[dict] | None:
    """The N=128 baseline rows from script 02 (this session or the bundled reference)."""
    hits: list[Path] = []
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if not root.exists():
            continue
        try:
            hits.extend(p for p in root.rglob("steer_baseline*.json") if p.is_file())
        except OSError:
            pass

    def _ok(p: Path) -> bool:
        s = str(p).replace("\\", "/").lower()
        # Must be a real N128 baseline, not this cell's own output.
        if "native_clean_k2" in p.name:
            return False
        if not re.fullmatch(r"steer_baseline(?: \(\d+\))?\.json", p.name):
            return False
        return "ncalib128" in s or "this_is_the_128" in s

    hits = [h for h in hits if _ok(h)]
    if not hits:
        print("no N128 reference baseline found; skipping the baseline agreement check")
        return None
    # Prefer this session's stage-2 output over the bundled reference copy.
    hits.sort(key=lambda p: (0 if "/kaggle/working" in str(p).replace("\\", "/") else 1, len(str(p))))
    print("reference N128 baseline ->", hits[0])
    return json.loads(hits[0].read_text(encoding="utf-8"))


def main() -> None:
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
    elapsed("pip install")

    import torch
    import transformers

    print("torch", torch.__version__, "transformers", transformers.__version__, "ngpu", torch.cuda.device_count())
    assert torch.cuda.is_available()

    repo = _bootstrap_repo()
    try:
        from huggingface_hub import login
        from kaggle_secrets import UserSecretsClient

        login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
        print("HF login ok")
    except Exception as e:
        print("HF login skipped:", type(e).__name__, e)
    elapsed("setup")

    OUT_DIR = Path("/kaggle/working/juice_stage2_int4_ncalib128")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    INT4_MODEL_ID_OVERRIDE = "/kaggle/working/llama31_8b_base_gptq_int4_ncalib128"

    HEADS_PATH = find_one("heads_int4_native_parametric_k2_clean.json")
    SPLIT_PATH = find_one("split.json")
    DATA_NAME = "conflictqa_pilot250_popular.jsonl"
    data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
    data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
    assert data_hits, "Missing conflictqa_pilot250_popular.jsonl"
    DATA_PATH = data_hits[0]
    print("DATA", DATA_PATH)

    from common.conflictqa import load_conflict_items
    from common.inference import greedy_generate
    from common.juice import (
        apply_juice_steering,
        conflict_eval_prompt,
        context_follow_rate,
        head_keys,
        load_identified_heads,
        wrap_llama_attention_for_juice,
    )
    from common.models import load_model_and_tokenizer, unload_model
    from common.scoring import label_conflict_response
    from common.stats_analysis import bootstrap_paired_follow_delta
    from tqdm import tqdm

    INT4_MODEL_ID = resolve_ncalib128_checkpoint(INT4_MODEL_ID_OVERRIDE)
    print("INT4_MODEL_ID", INT4_MODEL_ID)
    assert "ncalib128" in Path(INT4_MODEL_ID).name, INT4_MODEL_ID

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    eval_ids = set(split["eval_ids"])
    eval_items = [it for it in load_conflict_items(DATA_PATH) if it.id in eval_ids]
    assert len(eval_items) == split["n_eval"], (len(eval_items), split["n_eval"])
    assert split.get("seed") == 42, split.get("seed")
    print("n_eval", len(eval_items), "seed", split.get("seed"))

    ident = load_identified_heads(HEADS_PATH)
    print(TAG, "->", head_keys(ident.enlarge), "steer_target", ident.steer_target, "K", ident.K)
    assert head_keys(ident.enlarge) == EXPECTED_HEADS, head_keys(ident.enlarge)
    assert ident.K == 2 and ident.steer_target == "parametric"

    MAX_NEW_TOKENS = 16
    ALPHA_PLUS = 1.0
    ALPHA_MINUS = 0.0
    PROMPT_FORMAT = "juice_coherent"
    N_BOOT = 1000
    BOOT_SEED = 42

    # For context only. These heads have no FP16 counterpart (Jaccard 0.00 vs the
    # FP16 clean pair), so there is no same-sign gate to apply here.
    FP16_REF = {
        "baseline_fc": 0.16,
        "steer_parametric_k2_clean_fp16_pair": -8.0,
        "coherence": 1.0,
        "note": "FP16 base Cell-3 clean K2 = L20_H14/L26_H15, a DIFFERENT pair. Not a same-sign reference for L16_H30/L17_H27.",
    }
    NCALIB128_FP16_TRANSFER_CLEAN_K2_DELTA_PP = -4.0

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
    elapsed("model loaded")

    baseline = []
    for item in tqdm(eval_items, desc="baseline"):
        prompt = conflict_eval_prompt(tokenizer, item, prompt_format=PROMPT_FORMAT)
        text = greedy_generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
        lab = label_conflict_response(text, item.parametric_answer, item.context_answer)
        baseline.append({"id": item.id, "text": text, "label": lab, "follows_context": lab == "context"})

    r_base = context_follow_rate(baseline)
    c_base = coherence_rate(baseline)
    print(f"INT4 N128 baseline follows_context={r_base:.3f} coherence={c_base:.3f} counts={label_counts(baseline)}")
    show_examples(baseline, "INT4 N128 BASELINE (juice_coherent)")
    (OUT_DIR / "steer_baseline_native_clean_k2_cell.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    elapsed("baseline done")

    # Greedy decode on a fixed checkpoint should reproduce script 02's baseline.
    # Record agreement instead of asserting: a mismatch is a finding, not a crash.
    baseline_check: dict = {"reference_found": False}
    ref_rows = _load_reference_baseline()
    if ref_rows:
        ref_by = {r["id"]: bool(r.get("follows_context") or r.get("label") == "context") for r in ref_rows}
        cur_by = {r["id"]: bool(r["follows_context"]) for r in baseline}
        shared = sorted(set(ref_by) & set(cur_by))
        disagree = [i for i in shared if ref_by[i] != cur_by[i]]
        ref_fc = sum(ref_by[i] for i in shared) / max(1, len(shared))
        baseline_check = {
            "reference_found": True,
            "n_shared": len(shared),
            "reference_follows_context": ref_fc,
            "this_session_follows_context": r_base,
            "n_items_disagreeing": len(disagree),
            "ids_disagreeing": disagree,
            "identical": len(disagree) == 0,
        }
        print("baseline vs script-02 reference:", json.dumps({k: v for k, v in baseline_check.items() if k != "ids_disagreeing"}, indent=2))
        if disagree:
            print("WARNING: baseline differs from the script-02 N128 baseline on", len(disagree), "items")

    print("\n" + "=" * 70)
    print(TAG, "heads", head_keys(ident.enlarge))
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
        "fp16_delta_pp": None,
        "same_sign_as_fp16": None,
        "fp16_comparable": False,
        "id_source": "int4_base native directional clean filter (par_pos>0 AND dir_par>0), |D|=4",
        "par_pos_int4": ident.scores_pos,
        "ncalib128_fp16_transfer_clean_k2_delta_pp": NCALIB128_FP16_TRANSFER_CLEAN_K2_DELTA_PP,
        "bootstrap": boot["delta"],
    }
    print(json.dumps({k: v for k, v in entry.items() if k != "bootstrap"}, indent=2))
    print(
        "bootstrap delta_pp",
        boot["delta"]["point_pp"],
        "CI",
        boot["delta"]["ci95_pp"],
        "excl0",
        boot["delta"]["excludes_zero"],
    )
    show_examples(rows, TAG)
    (OUT_DIR / f"{TAG}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elapsed("native clean K2 done")

    summary = {
        "cell": "int4_native_clean_k2_apply",
        "identify_on": "int4_base",
        "apply_to": "int4_base_ncalib128",
        "model": INT4_MODEL_ID,
        "n_calib": 128,
        "prompt_format": PROMPT_FORMAT,
        "alpha_plus": ALPHA_PLUS,
        "alpha_minus": ALPHA_MINUS,
        "n_eval": len(eval_items),
        "eval_split_seed": split.get("seed"),
        "baseline": {
            "follows_context": r_base,
            "coherence": c_base,
            "label_counts": label_counts(baseline),
        },
        "baseline_vs_ncalib128_reference": baseline_check,
        "conditions": {TAG: entry},
        "fp16_reference": FP16_REF,
        "note": (
            "Native INT4 clean-filter pair L16_H30/L17_H27 (Jaccard 0.00 vs the FP16 clean pair). "
            "Previously identified but never steered. No FP16 same-sign gate applies."
        ),
    }
    (OUT_DIR / "int4_native_clean_k2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    payload = {
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
        "note": (
            "Resample the 50 eval items with replacement. Delta = steered "
            "follows_context − baseline follows_context, paired on item id."
        ),
        "n_calib": 128,
        "checkpoint": INT4_MODEL_ID,
        "primary_conditions": [TAG],
        "int4": {
            "precision": "int4",
            "n_baseline": len(baseline),
            "n_calib": 128,
            "conditions": {TAG: boot},
        },
        "fp16": None,
    }
    (OUT_DIR / "int4_native_clean_k2_bootstrap.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("wrote", OUT_DIR / "int4_native_clean_k2_bootstrap.json")

    print("\n=== TAKEAWAY (native INT4 clean K2 on N_CALIB=128) ===")
    print(f"{'condition':40s} {'fc':>8s} {'delta':>8s} {'CI lo':>8s} {'CI hi':>8s} {'excl0':>7s} {'coh':>6s}")
    print(f"{'baseline':40s} {r_base:8.3f} {0.0:8.1f} {'':>8s} {'':>8s} {'':>7s} {c_base:6.3f}")
    b = boot["delta"]
    print(
        f"{TAG:40s} {r:8.3f} {d_pp:8.1f} {b['ci95_pp'][0]:8.1f} {b['ci95_pp'][1]:8.1f} "
        f"{str(b['excludes_zero']):>7s} {coh:6.3f}"
    )
    print(
        "\nFor reference, transferred FP16 clean K2 (L20_H14/L26_H15) on this checkpoint: "
        f"{NCALIB128_FP16_TRANSFER_CLEAN_K2_DELTA_PP:+.1f} pp [-10.0, 0.0], includes 0."
    )
    print(
        "Reading rule: a usable parametric steer needs delta NEGATIVE with CI excluding 0 "
        "and `other` not exploding. Anything else is a null or a degradation, not a repair."
    )

    unload_model(model)
    elapsed("unload + done")
    print("done", OUT_DIR)


if __name__ == "__main__":
    main()
