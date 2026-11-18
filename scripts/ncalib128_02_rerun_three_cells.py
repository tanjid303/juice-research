"""Re-run the three locked Stage 2 INT4 cells against the N_CALIB=128 checkpoint.

Cells (same 50 eval ids, same seeds, same JuICE dual-run port):
  a) context K10 transfer — FP16 heads in heads_dir_context_k10.json
  b) random-K10 control — seed 42, exclude FP16 context K10 set
  c) parametric clean K2 — locked pair L20_H14, L26_H15

Writes int4_transfer_bootstrap.json with the same schema as
int_4_result/int4_transfer_bootstrap.json (plus random_context_k10).
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


def key_to_ref(k: str):
    from common.juice import HeadRef

    layer, head = k.replace("L", "").split("_H")
    return HeadRef(int(layer), int(head))


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

    # Pin to the N128 dir. Leave None to auto-detect *only* ..._ncalib128 folders.
    INT4_MODEL_ID_OVERRIDE = "/kaggle/working/llama31_8b_base_gptq_int4_ncalib128"

    print("=== /kaggle/input (top-level) ===")
    inp = Path("/kaggle/input")
    if inp.exists():
        for p in sorted(inp.iterdir()):
            print(" ", "DIR " if p.is_dir() else "FILE", p)
    else:
        print("  MISSING — attach datasets in the notebook sidebar")

    HEADS = {
        "ctx_k10": find_one("heads_dir_context_k10.json"),
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
    from common.inference import greedy_generate
    from common.juice import (
        apply_juice_steering,
        conflict_eval_prompt,
        context_follow_rate,
        head_keys,
        identified_from_enlarge,
        load_identified_heads,
        model_dims,
        random_head_control,
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
    print("n_eval", len(eval_items), "seed", split.get("seed"))
    assert split.get("seed") == 42, split.get("seed")

    ident_ctx = load_identified_heads(HEADS["ctx_k10"])
    ident_par = load_identified_heads(HEADS["par_k2_clean"])
    print("ctx_k10 ->", head_keys(ident_ctx.enlarge))
    print("par_k2_clean ->", head_keys(ident_par.enlarge))
    assert head_keys(ident_par.enlarge) == ["L20_H14", "L26_H15"], head_keys(ident_par.enlarge)

    MAX_NEW_TOKENS = 16
    ALPHA_PLUS = 1.0
    ALPHA_MINUS = 0.0
    PROMPT_FORMAT = "juice_coherent"
    N_BOOT = 1000
    BOOT_SEED = 42

    FP16_REF = {
        "baseline_fc": 0.16,
        "steer_context_k10": 10.0,
        "steer_parametric_k2_clean": -8.0,
        "coherence": 1.0,
        "note": "FP16 base Cell-3, juice_coherent, alpha+=1 alpha-=0, n=50",
    }

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
    (OUT_DIR / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    elapsed("baseline done")

    summary = {
        "cell": "fp16_to_int4_transfer",
        "identify_on": "fp16_base",
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
        "conditions": {},
        "fp16_reference": FP16_REF,
    }

    # --- (a) context K10 transfer ---
    print("\n" + "=" * 70)
    print("steer_context_k10 heads", head_keys(ident_ctx.enlarge))
    print("=" * 70)
    rows_ctx = apply_juice_steering(
        model,
        tokenizer,
        eval_items,
        ident_ctx,
        alpha_plus=ALPHA_PLUS,
        alpha_minus=ALPHA_MINUS,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_format=PROMPT_FORMAT,
    )
    r_ctx = context_follow_rate(rows_ctx)
    coh_ctx = coherence_rate(rows_ctx)
    d_ctx = (r_ctx - r_base) * 100
    entry_ctx = {
        "heads": head_keys(ident_ctx.enlarge),
        "steer_target": ident_ctx.steer_target,
        "follows_context": r_ctx,
        "delta_pp_vs_baseline": d_ctx,
        "coherence": coh_ctx,
        "label_counts": label_counts(rows_ctx),
        "fp16_delta_pp": FP16_REF["steer_context_k10"],
        "same_sign_as_fp16": (d_ctx == 0 and FP16_REF["steer_context_k10"] == 0)
        or (d_ctx * FP16_REF["steer_context_k10"] > 0),
    }
    summary["conditions"]["steer_context_k10"] = entry_ctx
    print(json.dumps(entry_ctx, indent=2))
    show_examples(rows_ctx, "steer_context_k10")
    (OUT_DIR / "steer_context_k10.json").write_text(json.dumps(rows_ctx, indent=2), encoding="utf-8")
    elapsed("context K10 done")

    # --- (b) random-K10 control (same seed/exclusion as native-ID notebook) ---
    n_layers, n_heads, _ = model_dims(model)
    print("dims", n_layers, n_heads)
    exclude_ctx = [key_to_ref(k) for k in head_keys(ident_ctx.enlarge)]
    rand_heads = random_head_control(n_layers, n_heads, 10, seed=42, exclude=exclude_ctx)
    ident_rand = identified_from_enlarge(
        rand_heads, precision="int4_random", K=10, steer_target="context"
    )
    print("\n" + "=" * 70)
    print("random_context_k10 heads", head_keys(ident_rand.enlarge), "seed=42")
    print("=" * 70)
    rows_rand = apply_juice_steering(
        model,
        tokenizer,
        eval_items,
        ident_rand,
        alpha_plus=ALPHA_PLUS,
        alpha_minus=ALPHA_MINUS,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_format=PROMPT_FORMAT,
    )
    r_rand = context_follow_rate(rows_rand)
    coh_rand = coherence_rate(rows_rand)
    d_rand = (r_rand - r_base) * 100
    boot_rand = bootstrap_paired_follow_delta(baseline, rows_rand, n_boot=N_BOOT, seed=BOOT_SEED)
    entry_rand = {
        "heads": head_keys(ident_rand.enlarge),
        "steer_target": "context",
        "K": 10,
        "seed": 42,
        "follows_context": r_rand,
        "delta_pp_vs_baseline": d_rand,
        "coherence": coh_rand,
        "label_counts": label_counts(rows_rand),
        "identified_transfer_delta_pp": entry_ctx["delta_pp_vs_baseline"],
        "bootstrap": boot_rand["delta"],
    }
    print(json.dumps({k: v for k, v in entry_rand.items() if k != "bootstrap"}, indent=2))
    print(
        "bootstrap delta_pp",
        boot_rand["delta"]["point_pp"],
        "CI",
        boot_rand["delta"]["ci95_pp"],
        "excl0",
        boot_rand["delta"]["excludes_zero"],
    )
    show_examples(rows_rand, "random_context_k10")
    (OUT_DIR / "random_context_k10.json").write_text(json.dumps(rows_rand, indent=2), encoding="utf-8")
    elapsed("random-K10 done")

    random_summary = {
        "cell": "int4_random_k_control",
        "apply_to": "int4_base_ncalib128",
        "model": INT4_MODEL_ID,
        "n_calib": 128,
        "prompt_format": PROMPT_FORMAT,
        "alpha_plus": ALPHA_PLUS,
        "alpha_minus": ALPHA_MINUS,
        "n_eval": len(eval_items),
        "baseline": {
            "follows_context": r_base,
            "coherence": c_base,
            "label_counts": label_counts(baseline),
        },
        "conditions": {"random_context_k10": entry_rand},
    }
    (OUT_DIR / "int4_random_k_summary.json").write_text(
        json.dumps(random_summary, indent=2), encoding="utf-8"
    )

    # --- (c) parametric clean K2 ---
    print("\n" + "=" * 70)
    print("steer_parametric_k2_clean heads", head_keys(ident_par.enlarge))
    print("=" * 70)
    rows_par = apply_juice_steering(
        model,
        tokenizer,
        eval_items,
        ident_par,
        alpha_plus=ALPHA_PLUS,
        alpha_minus=ALPHA_MINUS,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_format=PROMPT_FORMAT,
    )
    r_par = context_follow_rate(rows_par)
    coh_par = coherence_rate(rows_par)
    d_par = (r_par - r_base) * 100
    entry_par = {
        "heads": head_keys(ident_par.enlarge),
        "steer_target": ident_par.steer_target,
        "follows_context": r_par,
        "delta_pp_vs_baseline": d_par,
        "coherence": coh_par,
        "label_counts": label_counts(rows_par),
        "fp16_delta_pp": FP16_REF["steer_parametric_k2_clean"],
        "same_sign_as_fp16": (d_par == 0 and FP16_REF["steer_parametric_k2_clean"] == 0)
        or (d_par * FP16_REF["steer_parametric_k2_clean"] > 0),
    }
    summary["conditions"]["steer_parametric_k2_clean"] = entry_par
    print(json.dumps(entry_par, indent=2))
    show_examples(rows_par, "steer_parametric_k2_clean")
    (OUT_DIR / "steer_parametric_k2_clean.json").write_text(json.dumps(rows_par, indent=2), encoding="utf-8")
    elapsed("parametric clean K2 done")

    (OUT_DIR / "int4_transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Paired bootstrap — same function / n_boot / seed as scripts/bootstrap_int4_transfer.py
    print("=== paired bootstrap n=50, 1000 draws, seed=42 ===")
    boot_ctx = bootstrap_paired_follow_delta(baseline, rows_ctx, n_boot=N_BOOT, seed=BOOT_SEED)
    boot_par = bootstrap_paired_follow_delta(baseline, rows_par, n_boot=N_BOOT, seed=BOOT_SEED)
    for tag, boot in (
        ("steer_context_k10", boot_ctx),
        ("random_context_k10", boot_rand),
        ("steer_parametric_k2_clean", boot_par),
    ):
        d = boot["delta"]
        print(
            f"int4  {tag:32s}  d={d['point_pp']:+6.1f} pp  "
            f"CI95=[{d['ci95_pp'][0]:+6.1f}, {d['ci95_pp'][1]:+6.1f}]  "
            f"excludes0={d['excludes_zero']}"
        )

    NOTE = (
        "Resample the 50 eval items with replacement. Delta = steered "
        "follows_context − baseline follows_context, paired on item id."
    )
    fp16_payload = None
    n64_boot = None
    n64_hits: list[Path] = []
    for root in (repo, Path("/kaggle/working"), Path("/kaggle/input")):
        if not root.exists():
            continue
        try:
            n64_hits.extend(
                p
                for p in root.rglob("int4_transfer_bootstrap.json")
                if "n64_baseline" in str(p).replace("\\", "/")
            )
        except OSError:
            pass
    n64_hits = [h for h in n64_hits if h.is_file()]
    if n64_hits:
        n64_boot = json.loads(n64_hits[0].read_text(encoding="utf-8"))
        print("N64 bootstrap (for fp16 block) ->", n64_hits[0])
    if n64_boot is not None:
        fp16_payload = n64_boot.get("fp16")

    payload = {
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
        "note": NOTE,
        "n_calib": 128,
        "checkpoint": INT4_MODEL_ID,
        "primary_conditions": [
            "steer_context_k10",
            "random_context_k10",
            "steer_parametric_k2_clean",
        ],
        "int4": {
            "precision": "int4",
            "n_baseline": len(baseline),
            "n_calib": 128,
            "conditions": {
                "steer_context_k10": boot_ctx,
                "random_context_k10": boot_rand,
                "steer_parametric_k2_clean": boot_par,
            },
        },
        "fp16": fp16_payload,
    }
    (OUT_DIR / "int4_transfer_bootstrap.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("wrote", OUT_DIR / "int4_transfer_bootstrap.json")

    print("\n=== TAKEAWAY TABLE (INT4 N_CALIB=128) ===")
    print(f"{'condition':32s} {'int4_fc':>8s} {'int4_d':>8s} {'fp16_d':>8s} {'sign':>6s} {'coh':>6s}")
    print(f"{'baseline':32s} {r_base:8.3f} {0.0:8.1f} {0.0:8.1f} {'':6s} {c_base:6.3f}")
    for tag, entry in summary["conditions"].items():
        sign = "OK" if entry["same_sign_as_fp16"] else "FLIP"
        print(
            f"{tag:32s} {entry['follows_context']:8.3f} "
            f"{entry['delta_pp_vs_baseline']:8.1f} {entry['fp16_delta_pp']:8.1f} "
            f"{sign:6s} {entry['coherence']:6.3f}"
        )
    print(
        f"{'random_context_k10':32s} {r_rand:8.3f} {d_rand:8.1f} {'n/a':>8s} "
        f"{'':6s} {coh_rand:6.3f}"
    )

    unload_model(model)
    elapsed("unload + done")
    print("done", OUT_DIR)


if __name__ == "__main__":
    main()
