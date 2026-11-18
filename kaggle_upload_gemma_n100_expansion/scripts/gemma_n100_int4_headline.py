"""Transfer FP16 L27_H6 (K=1) onto Gemma-2-9B GPTQ N_CALIB=128.

Same 100 eval ids as Gemma ID / Cell-3, juice_coherent, greedy, α+=1, α−=0,
paired bootstrap 1000 / seed 42, plus a random-K1 control (seed 42, exclude L27_H6).

Does not requantize. Requires scripts/gemma2_02_quant_smoke_test.py to have passed
in this session, or a known-good checkpoint.
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
    assert cands, "Attach kaggle_juice_gemma2_int4_transfer_l27h6.zip"

    def _ok(models_py: Path) -> bool:
        ja = models_py.parent / "juice_attn.py"
        if not ja.is_file():
            return False
        txt = ja.read_text(encoding="utf-8", errors="ignore")
        return "GetHeadsGemma2Attention" in txt and "dense_linear_weight_T" in txt

    cands.sort(key=lambda p: (_ok(p), "gemma2" in str(p).lower()), reverse=True)
    assert _ok(cands[0]), "common/juice_attn.py missing Gemma GPTQ o_proj dequant"
    repo = cands[0].parent.parent
    dst_common = Path("/kaggle/working/common")
    if not dst_common.exists() or not (dst_common / "juice_attn.py").is_file():
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
    hits.sort(key=lambda p: (0 if "gemma_n100_expansion" in str(p).lower() or "gemma_n100" in str(p).lower() else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]


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


def show_examples(rows, title, n_show=5):
    print(f"\n===== {title}: {n_show} raw examples =====")
    for r in rows[:n_show]:
        print(f"- {r['id']} [{r['label']}] {r['text']!r}")


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
            "huggingface-hub",
            "tqdm",
            "numpy",
            "pandas",
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

    OUT_DIR = Path("/kaggle/working/gemma_n100_int4_headline")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from common.conflictqa import load_conflict_items
    from common.inference import greedy_generate
    from common.juice import (
        HeadRef,
        apply_juice_steering,
        conflict_eval_prompt,
        context_follow_rate,
        head_keys,
        identified_from_enlarge,
        load_identified_heads,
        model_dims,
        random_head_control,
        wrap_attention_for_juice,
    )
    from common.models import load_model_and_tokenizer, resolve_gemma2_ncalib128_checkpoint, unload_model
    from common.scoring import label_conflict_response
    from common.stats_analysis import bootstrap_paired_follow_delta
    from tqdm import tqdm

    INT4_MODEL_ID = resolve_gemma2_ncalib128_checkpoint()
    print("INT4_MODEL_ID", INT4_MODEL_ID)
    assert "ncalib128" in Path(INT4_MODEL_ID).name, INT4_MODEL_ID

    RAW_HEADS_PATH = find_one("heads_dir_parametric_k1_raw.json")
    CTX_HEADS_PATH = find_one("heads_dir_context_k10.json")
    split_candidates = []
    for _root in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if _root.exists():
            split_candidates.extend(_root.rglob("split.json"))
    SPLIT_PATH = next(
        (p for p in split_candidates
         if json.loads(p.read_text(encoding="utf-8")).get("n_eval") == 100),
        None,
    )
    assert SPLIT_PATH is not None, "Missing n=100 split.json"
    DATA_NAME = "conflictqa_pilot250_popular.jsonl"
    data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
    data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
    assert data_hits, DATA_NAME
    DATA_PATH = data_hits[0]
    print("DATA", DATA_PATH)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    eval_ids = set(split["eval_ids"])
    eval_items = [it for it in load_conflict_items(DATA_PATH) if it.id in eval_ids]
    assert len(eval_items) == split["n_eval"] == 100, (len(eval_items), split["n_eval"])
    assert split.get("seed") == 42, split.get("seed")
    print("n_eval", len(eval_items), "seed", split.get("seed"))

    ident = load_identified_heads(RAW_HEADS_PATH)
    ident_ctx = load_identified_heads(CTX_HEADS_PATH)
    assert head_keys(ident.enlarge) == ["L27_H6"], head_keys(ident.enlarge)
    assert len(ident_ctx.enlarge) == 10, head_keys(ident_ctx.enlarge)
    print("steer_parametric_k1_raw", head_keys(ident.enlarge))

    MAX_NEW_TOKENS = 16
    ALPHA_PLUS = 1.0
    ALPHA_MINUS = 0.0
    PROMPT_FORMAT = "juice_coherent"
    N_BOOT = 1000
    BOOT_SEED = 42
    RAND_SEED = 42

    FP16_REF = {
        "baseline_fc": 0.48,
        "steer_parametric_k1_raw": -12.0,
        "coherence": 1.0,
        "ci95_pp": [-22.0, -4.0],
        "excludes_zero": True,
        "label_counts": {"parametric": 28, "context": 18, "rejects_context": 0, "other": 4},
        "note": "Gemma-2-9B FP16 Cell-3, juice_coherent, alpha+=1 alpha-=0, n=100, L27_H6 K=1",
    }

    print("Loading INT4", INT4_MODEL_ID)
    model, tokenizer = load_model_and_tokenizer(
        "int4",
        model_id_int4=INT4_MODEL_ID,
        device_map={"": 0},
    )
    print("device_map", getattr(model, "hf_device_map", None))
    print("model_type", model.config.model_type)
    assert model.config.model_type == "gemma2"
    wrap_attention_for_juice(model)
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
    print(
        f"INT4 N128 baseline fc={r_base:.3f} coh={c_base:.3f} "
        f"P/C/O={pco(label_counts(baseline))} counts={label_counts(baseline)}"
    )
    show_examples(baseline, "INT4 N128 BASELINE (juice_coherent)")
    (OUT_DIR / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    elapsed("baseline done")

    n_layers, n_heads, head_dim = model_dims(model)
    print("dims", n_layers, n_heads, head_dim)
    exclude = list(ident.enlarge)
    rand_heads = random_head_control(n_layers, n_heads, 1, seed=RAND_SEED, exclude=exclude)
    ident_rand = identified_from_enlarge(
        rand_heads, precision="int4_gemma2_random", K=1, steer_target="parametric"
    )
    assert head_keys(ident_rand.enlarge) != ["L27_H6"]
    print("random_parametric_k1", head_keys(ident_rand.enlarge), "seed", RAND_SEED, "excluded", ["L27_H6"])
    (OUT_DIR / "random_k1_heads.json").write_text(
        json.dumps(
            {"heads": head_keys(ident_rand.enlarge), "seed": RAND_SEED, "excluded": ["L27_H6"]},
            indent=2,
        ),
        encoding="utf-8",
    )

    ctx_rand_heads = random_head_control(
        n_layers, n_heads, 10, seed=RAND_SEED, exclude=list(ident_ctx.enlarge)
    )
    ident_ctx_rand = identified_from_enlarge(
        ctx_rand_heads, precision="int4_gemma2_random_context", K=10,
        steer_target="context"
    )
    (OUT_DIR / "random_k10_heads.json").write_text(
        json.dumps({"heads": head_keys(ctx_rand_heads), "seed": RAND_SEED,
                    "excluded": head_keys(ident_ctx.enlarge)}, indent=2),
        encoding="utf-8"
    )
    conditions = [
        ("steer_context_k10", ident_ctx),
        ("random_context_k10", ident_ctx_rand),
        ("steer_parametric_k1_raw", ident),
        ("random_parametric_k1", ident_rand),
    ]
    summary = {
        "cell": "fp16_to_int4_transfer_l27h6",
        "identify_on": "fp16_gemma2_base",
        "apply_to": "int4_gemma2_base_ncalib128",
        "model": INT4_MODEL_ID,
        "n_calib": 128,
        "prompt_format": PROMPT_FORMAT,
        "alpha_plus": ALPHA_PLUS,
        "alpha_minus": ALPHA_MINUS,
        "n_eval": len(eval_items),
        "eval_split_seed": split.get("seed"),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "random_k_seed": RAND_SEED,
        "baseline": {
            "follows_context": r_base,
            "coherence": c_base,
            "label_counts": label_counts(baseline),
        },
        "conditions": {},
        "fp16_reference": FP16_REF,
    }

    row_store = {}
    for tag, ident_i in conditions:
        print("\n" + "=" * 70)
        print(tag, "heads", head_keys(ident_i.enlarge))
        print("=" * 70)
        rows = apply_juice_steering(
            model,
            tokenizer,
            eval_items,
            ident_i,
            alpha_plus=ALPHA_PLUS,
            alpha_minus=ALPHA_MINUS,
            max_new_tokens=MAX_NEW_TOKENS,
            prompt_format=PROMPT_FORMAT,
        )
        r = context_follow_rate(rows)
        coh = coherence_rate(rows)
        d_pp = (r - r_base) * 100
        boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=N_BOOT, seed=BOOT_SEED)
        fp16_d = FP16_REF["steer_parametric_k1_raw"] if tag == "steer_parametric_k1_raw" else None
        entry = {
            "heads": head_keys(ident_i.enlarge),
            "steer_target": ident_i.steer_target,
            "K": ident_i.K,
            "follows_context": r,
            "delta_pp_vs_baseline": d_pp,
            "coherence": coh,
            "label_counts": label_counts(rows),
            "bootstrap": boot["delta"],
        }
        if fp16_d is not None:
            entry["fp16_delta_pp"] = fp16_d
            entry["same_sign_as_fp16"] = (d_pp == 0 and fp16_d == 0) or (d_pp * fp16_d > 0)
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

    summary_path = OUT_DIR / "gemma2_int4_transfer_l27h6_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    NOTE = (
        "Resample the 100 eval items with replacement. Delta = steered "
        "follows_context − baseline follows_context, paired on item id."
    )
    fp16_payload = None
    fp16_hits = list(Path("/kaggle/working").rglob("gemma2_base_cell3_bootstrap.json"))
    fp16_hits += list(Path("/kaggle/input").rglob("gemma2_base_cell3_bootstrap.json")) if Path("/kaggle/input").exists() else []
    fp16_hits = [p for p in fp16_hits if p.is_file()]
    if fp16_hits:
        raw = json.loads(fp16_hits[0].read_text(encoding="utf-8"))
        fp16_payload = {
            "precision": "fp16",
            "n_baseline": 100,
            "source": str(fp16_hits[0]),
            "conditions": {
                "steer_parametric_k1_raw": raw["fp16"]["conditions"]["steer_parametric_k1_raw"]
            },
        }
        print("fp16 reference bootstrap ->", fp16_hits[0])

    boot_payload = {
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
        "note": NOTE,
        "n_calib": 128,
        "checkpoint": INT4_MODEL_ID,
        "primary_conditions": ["steer_parametric_k1_raw", "random_parametric_k1"],
        "int4": {
            "precision": "int4",
            "n_baseline": len(baseline),
            "n_calib": 128,
            "conditions": {
                tag: bootstrap_paired_follow_delta(baseline, row_store[tag], n_boot=N_BOOT, seed=BOOT_SEED)
                for tag in summary["conditions"]
            },
        },
    }
    if fp16_payload is not None:
        boot_payload["fp16"] = fp16_payload
    boot_path = OUT_DIR / "gemma2_int4_transfer_l27h6_bootstrap.json"
    boot_path.write_text(json.dumps(boot_payload, indent=2), encoding="utf-8")
    # Easy-download copies at working root.
    Path("/kaggle/working/gemma2_int4_transfer_l27h6_summary.json").write_text(
        summary_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    Path("/kaggle/working/gemma2_int4_transfer_l27h6_bootstrap.json").write_text(
        boot_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    print("\n=== GEMMA-2-9B INT4 TRANSFER (L27_H6) ===")
    hdr = f"{'condition':28s} {'fc':>6s} {'d_pp':>7s} {'CI95':>18s} {'excl0':>6s} {'coh':>6s} {'P/C/O':>8s}"
    print(hdr)
    print("-" * len(hdr))
    print(
        f"{'baseline':28s} {r_base:6.3f} {0.0:7.1f} {'—':>18s} {'—':>6s} "
        f"{c_base:6.3f} {pco(label_counts(baseline)):>8s}"
    )
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
    steered = summary["conditions"]["steer_parametric_k1_raw"]
    print(
        f"\nFP16 L27_H6 was {FP16_REF['steer_parametric_k1_raw']:+.1f} pp "
        f"CI {FP16_REF['ci95_pp']}. INT4 same-sign={steered.get('same_sign_as_fp16')}"
    )
    unload_model(model)
    print()
    print("=" * 60)
    print("SAVE VERSION NOW — Quick Save, not Save & Run All")
    print("Need:", summary_path.name, "and", boot_path.name)
    print("folder:", OUT_DIR)
    print("=" * 60)
    elapsed("done")
    print("repo", repo)


if __name__ == "__main__":
    main()
