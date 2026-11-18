"""Transfer FP16 context-K10 suppress (H−) onto Gemma-2-9B GPTQ N_CALIB=128.

FP16 enlarge of this set was a null (ceiling). FP16 suppress was −12 pp, CI
excludes 0. This cell tests whether that suppress recipe survives GPTQ.

Does not requantize. Same 50 eval ids, juice_coherent, greedy, α+=0, α−=1,
paired bootstrap 1000 / seed 42. Random-K10 suppress control reuses the
Gemma FP16 Cell-3 random-K10 enlarge draw (seed 42).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

STAGE_T0 = time.time()


def elapsed(stage: str) -> None:
    print(f"[elapsed] {stage}: {(time.time() - STAGE_T0) / 60:.1f} min since start")


def _bootstrap_repo() -> Path:
    cands = list(Path("/kaggle/working").rglob("common/models.py"))
    cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
    assert cands, "Attach kaggle_juice_gemma2_int4_context_suppress.zip"

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
    hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() else 1, len(str(p))))
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


def key_to_ref(k: str):
    from common.juice import HeadRef

    layer, head = k.replace("L", "").split("_H")
    return HeadRef(int(layer), int(head))


def ident_suppress(heads, *, precision: str, K: int):
    from common.juice import IdentifiedHeads

    return IdentifiedHeads(
        enlarge=[],
        suppress=list(heads),
        scores_pos={},
        scores_neg={},
        precision=precision,
        K=K,
        steer_target="context",
        n_calibration=0,
        pos_alphas=[],
        neg_alphas=[],
    )


def flip_report(baseline, steered, title: str) -> dict:
    base = {r["id"]: r for r in baseline}
    trans = Counter()
    ctx_gain = ctx_loss = 0
    text_changed = 0
    flips = []
    for r in steered:
        b = base[r["id"]]
        trans[(b["label"], r["label"])] += 1
        if b["text"] != r["text"]:
            text_changed += 1
        if b["label"] != "context" and r["label"] == "context":
            ctx_gain += 1
        if b["label"] == "context" and r["label"] != "context":
            ctx_loss += 1
        if b["label"] != r["label"]:
            flips.append(
                {
                    "id": r["id"],
                    "from": b["label"],
                    "to": r["label"],
                    "base": b["text"],
                    "steered": r["text"],
                }
            )
    other_b = sum(1 for r in baseline if r["label"] == "other")
    other_s = sum(1 for r in steered if r["label"] == "other")
    payload = {
        "n_label_flips": len(flips),
        "text_changed": text_changed,
        "ctx_gain": ctx_gain,
        "ctx_loss": ctx_loss,
        "net_ctx": ctx_gain - ctx_loss,
        "transitions": {f"{a}->{b}": n for (a, b), n in trans.items() if a != b},
        "other_baseline": other_b,
        "other_steered": other_s,
        "other_delta": other_s - other_b,
        "other_inflated": other_s > other_b,
        "flips": flips,
    }
    print(f"\n=== FLIPS {title} ===")
    print(
        f"  label_flips={len(flips)} text_changed={text_changed} "
        f"net_ctx={ctx_gain - ctx_loss} (+{ctx_gain}/-{ctx_loss})"
    )
    for (a, b), n in sorted(trans.items(), key=lambda kv: -kv[1]):
        if a != b:
            print(f"  {a:16s} -> {b:16s} {n}")
    print(f"  other {other_b} -> {other_s}  inflated={other_s > other_b}")
    if other_s > other_b:
        print("  WARNING: other rose — Llama α=3/5 trap pattern, inspect raw text")
    for f in flips:
        print(f"  flip {f['id']} {f['from']}->{f['to']}")
        print(f"    base={f['base']!r}")
        print(f"    ste={f['steered']!r}")
    return payload


def main() -> None:
    print("THIS IS GEMMA2_05 — INT4 context K10 suppress transfer. Not a notebook generator.")
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

    OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_int4_context_suppress")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from common.conflictqa import load_conflict_items
    from common.inference import greedy_generate
    from common.juice import (
        apply_juice_steering,
        conflict_eval_prompt,
        context_follow_rate,
        head_keys,
        load_identified_heads,
        wrap_attention_for_juice,
    )
    from common.models import load_model_and_tokenizer, resolve_gemma2_ncalib128_checkpoint, unload_model
    from common.scoring import label_conflict_response
    from common.stats_analysis import bootstrap_paired_follow_delta
    from tqdm import tqdm

    INT4_MODEL_ID = resolve_gemma2_ncalib128_checkpoint()
    print("INT4_MODEL_ID", INT4_MODEL_ID)
    assert "ncalib128" in Path(INT4_MODEL_ID).name, INT4_MODEL_ID

    HEADS_PATH = find_one("heads_dir_context_k10.json")
    SPLIT_PATH = find_one("split.json")
    DATA_NAME = "conflictqa_pilot250_popular.jsonl"
    data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
    data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
    assert data_hits, DATA_NAME
    DATA_PATH = data_hits[0]
    print("DATA", DATA_PATH)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    eval_ids = set(split["eval_ids"])
    eval_items = [it for it in load_conflict_items(DATA_PATH) if it.id in eval_ids]
    assert len(eval_items) == split["n_eval"] == 50, (len(eval_items), split["n_eval"])
    assert split.get("seed") == 42, split.get("seed")
    print("n_eval", len(eval_items), "seed", split.get("seed"))

    ident_ctx = load_identified_heads(HEADS_PATH)
    ident_sup = ident_suppress(ident_ctx.enlarge, precision="fp16_gemma2_base", K=10)
    ctx_keys = head_keys(ident_sup.suppress)
    print("suppress_context_k10", ctx_keys)
    assert ident_sup.enlarge == []
    assert len(ident_sup.suppress) == 10

    RAND_NOTE = (
        "Reused Gemma FP16 Cell-3 random-K10 enlarge draw (seed 42). "
        "That draw excluded ctx K10 ∪ par K2 (L31_H7, L39_H6) ∪ L27_H6. "
        "Applied here as H− (α+=0, α−=1), not resampled."
    )
    rand_path_hits = list(Path("/kaggle/working").rglob("random_k10_heads.json"))
    rand_path_hits += list(Path("/kaggle/input").rglob("random_k10_heads.json")) if Path("/kaggle/input").exists() else []
    rand_path_hits = [p for p in rand_path_hits if p.is_file()]
    rand_path_hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() or "cell3" in str(p).lower() else 1, len(str(p))))
    if rand_path_hits:
        rand_payload = json.loads(rand_path_hits[0].read_text(encoding="utf-8"))
        rand_keys = list(rand_payload["heads"])
        print("random_k10 source", rand_path_hits[0])
        print(RAND_NOTE)
    else:
        print("WARNING: random_k10_heads.json missing; will resample seed=42 exclude ctx K10 only after load")
        RAND_NOTE = "Fallback resample seed=42, exclude identified context K10 only (Cell-3 draw missing)."
        rand_keys = None

    MAX_NEW_TOKENS = 16
    ALPHA_PLUS = 0.0
    ALPHA_MINUS = 1.0
    PROMPT_FORMAT = "juice_coherent"
    N_BOOT = 1000
    BOOT_SEED = 42

    FP16_REF = {
        "baseline_fc": 0.48,
        "suppress_context_k10": -12.0,
        "coherence": 1.0,
        "ci95_pp": [-24.0, -2.0],
        "excludes_zero": True,
        "label_counts": {"parametric": 32, "context": 18, "rejects_context": 0, "other": 0},
        "note": "Gemma-2-9B FP16 Task-2 suppress, juice_coherent, alpha+=0 alpha-=1, n=50",
    }

    print("Loading INT4 (no requantize)", INT4_MODEL_ID)
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

    if rand_keys is None:
        from common.juice import model_dims, random_head_control

        n_layers, n_heads, _ = model_dims(model)
        rand_heads = random_head_control(n_layers, n_heads, 10, seed=42, exclude=ident_sup.suppress)
        rand_keys = head_keys(rand_heads)

    overlap = set(rand_keys) & set(ctx_keys)
    assert not overlap, overlap
    ident_rand = ident_suppress([key_to_ref(k) for k in rand_keys], precision="int4_gemma2_random", K=10)
    print("random_suppress_k10", rand_keys)
    (OUT_DIR / "random_k10_heads.json").write_text(
        json.dumps(
            {
                "heads": rand_keys,
                "seed": 42,
                "excluded": ctx_keys,
                "source": RAND_NOTE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

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

    conditions = [
        ("suppress_context_k10", ident_sup),
        ("random_suppress_k10", ident_rand),
    ]
    summary = {
        "cell": "fp16_to_int4_transfer_context_k10_suppress",
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
        "random_k_seed": 42,
        "random_k_source": RAND_NOTE,
        "baseline": {
            "follows_context": r_base,
            "coherence": c_base,
            "label_counts": label_counts(baseline),
        },
        "conditions": {},
        "fp16_reference": FP16_REF,
    }

    row_store = {}
    flip_store = {}
    for tag, ident_i in conditions:
        print("\n" + "=" * 70)
        print(tag, "H- heads", head_keys(ident_i.suppress), "enlarge", head_keys(ident_i.enlarge))
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
        flips = flip_report(baseline, rows, tag)
        flip_store[tag] = {k: v for k, v in flips.items() if k != "flips"}
        entry = {
            "heads": head_keys(ident_i.suppress),
            "steer_target": "context_suppress",
            "K": ident_i.K,
            "alpha_plus": ALPHA_PLUS,
            "alpha_minus": ALPHA_MINUS,
            "follows_context": r,
            "delta_pp_vs_baseline": d_pp,
            "coherence": coh,
            "label_counts": label_counts(rows),
            "bootstrap": boot["delta"],
            "flips": flip_store[tag],
        }
        if tag == "suppress_context_k10":
            entry["fp16_delta_pp"] = FP16_REF["suppress_context_k10"]
            entry["same_sign_as_fp16"] = (d_pp == 0 and FP16_REF["suppress_context_k10"] == 0) or (
                d_pp * FP16_REF["suppress_context_k10"] > 0
            )
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

    summary_path = OUT_DIR / "gemma2_int4_context_suppress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    NOTE = (
        "Resample the 50 eval items with replacement. Delta = steered "
        "follows_context − baseline follows_context, paired on item id."
    )
    fp16_payload = None
    fp16_hits = list(Path("/kaggle/working").rglob("gemma2_fp16_context_suppress_bootstrap.json"))
    fp16_hits += (
        list(Path("/kaggle/input").rglob("gemma2_fp16_context_suppress_bootstrap.json"))
        if Path("/kaggle/input").exists()
        else []
    )
    fp16_hits = [p for p in fp16_hits if p.is_file()]
    if fp16_hits:
        raw = json.loads(fp16_hits[0].read_text(encoding="utf-8"))
        fp16_payload = {
            "precision": "fp16",
            "n_baseline": 50,
            "source": str(fp16_hits[0]),
            "conditions": {
                "suppress_context_k10": raw["fp16"]["conditions"]["suppress_context_k10"]
            },
        }
        print("fp16 suppress bootstrap ->", fp16_hits[0])

    boot_payload = {
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
        "note": NOTE,
        "n_calib": 128,
        "checkpoint": INT4_MODEL_ID,
        "primary_conditions": ["suppress_context_k10", "random_suppress_k10"],
        "random_k_source": RAND_NOTE,
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
    boot_path = OUT_DIR / "gemma2_int4_context_suppress_bootstrap.json"
    boot_path.write_text(json.dumps(boot_payload, indent=2), encoding="utf-8")
    Path("/kaggle/working/gemma2_int4_context_suppress_summary.json").write_text(
        summary_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    Path("/kaggle/working/gemma2_int4_context_suppress_bootstrap.json").write_text(
        boot_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    print("\n=== GEMMA-2-9B INT4 CONTEXT K10 SUPPRESS TRANSFER ===")
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
    steered = summary["conditions"]["suppress_context_k10"]
    print(
        f"\nFP16 suppress was {FP16_REF['suppress_context_k10']:+.1f} pp "
        f"CI {FP16_REF['ci95_pp']}. INT4 same-sign={steered.get('same_sign_as_fp16')}"
    )
    print("Random-K10:", RAND_NOTE)
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
