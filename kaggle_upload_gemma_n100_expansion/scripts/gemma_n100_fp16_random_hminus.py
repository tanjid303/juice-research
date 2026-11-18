"""FP16 random-K10 suppress control on Gemma-2-9B base.

Applies the same 10 heads as the INT4 suppress random control (Gemma Cell-3
enlarge random-K10 draw, seed 42) as H− (α+=0, α−=1, enlarge empty).

Tests whether the FP16 identified ctx-K10 suppress (−12 pp, CI excludes 0)
was head-specific, or whether generic 10-head H− at this α moves fc.
No INT4 checkpoint.
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

EXPECTED_RANDOM_K10 = [
    "L41_H11",
    "L7_H3",
    "L1_H9",
    "L17_H13",
    "L15_H13",
    "L14_H7",
    "L9_H1",
    "L6_H9",
    "L35_H9",
    "L5_H10",
]


def elapsed(stage: str) -> None:
    print(f"[elapsed] {stage}: {(time.time() - STAGE_T0) / 60:.1f} min since start")


def _bootstrap_repo() -> Path:
    cands = list(Path("/kaggle/working").rglob("common/models.py"))
    cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
    assert cands, "Attach kaggle_juice_gemma2_fp16_random_k10_suppress.zip"

    def _ok(models_py: Path) -> bool:
        ja = models_py.parent / "juice_attn.py"
        if not ja.is_file():
            return False
        return "GetHeadsGemma2Attention" in ja.read_text(encoding="utf-8", errors="ignore")

    cands.sort(key=lambda p: (_ok(p), "gemma2" in str(p).lower()), reverse=True)
    assert _ok(cands[0])
    repo = cands[0].parent.parent
    dst_common = Path("/kaggle/working/common")
    if not dst_common.exists() or not (dst_common / "juice.py").is_file():
        if dst_common.exists():
            shutil.rmtree(dst_common)
        shutil.copytree(repo / "common", dst_common)
    if "/kaggle/working" not in sys.path:
        sys.path.insert(0, "/kaggle/working")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    print("REPO", repo)
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


def key_to_ref(k: str):
    from common.juice import HeadRef

    layer, head = k.replace("L", "").split("_H")
    return HeadRef(int(layer), int(head))


def ident_suppress(keys):
    from common.juice import IdentifiedHeads

    heads = [key_to_ref(k) for k in keys]
    return IdentifiedHeads(
        enlarge=[],
        suppress=heads,
        scores_pos={},
        scores_neg={},
        precision="fp16_gemma2_random",
        K=10,
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
                {"id": r["id"], "from": b["label"], "to": r["label"], "base": b["text"], "steered": r["text"]}
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
    for f in flips:
        print(f"  flip {f['id']} {f['from']}->{f['to']}")
        print(f"    base={f['base']!r}")
        print(f"    ste={f['steered']!r}")
    return payload


def main() -> None:
    print("THIS IS GEMMA2_06 — FP16 random-K10 suppress. Not a notebook generator.")
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
    from huggingface_hub import login
    from kaggle_secrets import UserSecretsClient

    login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
    print("HF login ok")
    elapsed("setup")

    OUT_DIR = Path("/kaggle/working/gemma_n100_fp16_random_hminus")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ID = "google/gemma-2-9b"

    from common.conflictqa import load_conflict_items
    from common.inference import greedy_generate
    from common.juice import apply_juice_steering, conflict_eval_prompt, context_follow_rate, head_keys
    from common.models import gpu_max_memory, load_model_and_tokenizer, unload_model
    from common.scoring import label_conflict_response
    from common.stats_analysis import bootstrap_paired_follow_delta
    from tqdm import tqdm

    RAND_PATH = find_one("random_k10_heads.json")
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

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    eval_ids = set(split["eval_ids"])
    eval_items = [it for it in load_conflict_items(DATA_PATH) if it.id in eval_ids]
    assert len(eval_items) == split["n_eval"] == 100
    assert split.get("seed") == 42
    print("n_eval", len(eval_items), "seed", split.get("seed"))

    rand_payload = json.loads(RAND_PATH.read_text(encoding="utf-8"))
    rand_keys = list(rand_payload["heads"])
    assert rand_keys == EXPECTED_RANDOM_K10, (rand_keys, EXPECTED_RANDOM_K10)
    RAND_NOTE = (
        "Same Cell-3 enlarge random-K10 draw (seed 42) used as the INT4 suppress "
        "random control. Applied here as H− on FP16. Not resampled."
    )
    print("random_k10", rand_keys)
    print(RAND_NOTE)

    ident_rand = ident_suppress(rand_keys)
    assert ident_rand.enlarge == []
    assert head_keys(ident_rand.suppress) == EXPECTED_RANDOM_K10

    MAX_NEW_TOKENS = 16
    ALPHA_PLUS = 0.0
    ALPHA_MINUS = 1.0
    PROMPT_FORMAT = "juice_coherent"
    N_BOOT = 1000
    BOOT_SEED = 42

    FP16_IDENTIFIED_REF = {
        "suppress_context_k10": -12.0,
        "ci95_pp": [-24.0, -2.0],
        "excludes_zero": True,
        "note": "FP16 Task-2 identified ctx K10 H−",
    }

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
        f"FP16 baseline fc={r_base:.3f} coh={c_base:.3f} "
        f"P/C/O={pco(label_counts(baseline))} counts={label_counts(baseline)}"
    )
    show_examples(baseline, "FP16 BASELINE (juice_coherent)")
    (OUT_DIR / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    elapsed("baseline")

    print("\n" + "=" * 70)
    print("random_suppress_k10  alpha+=0 alpha-=1  heads", rand_keys)
    print("=" * 70)
    rows = apply_juice_steering(
        model,
        tokenizer,
        eval_items,
        ident_rand,
        alpha_plus=ALPHA_PLUS,
        alpha_minus=ALPHA_MINUS,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_format=PROMPT_FORMAT,
    )
    r = context_follow_rate(rows)
    coh = coherence_rate(rows)
    d_pp = (r - r_base) * 100
    boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=N_BOOT, seed=BOOT_SEED)
    flips = flip_report(baseline, rows, "random_suppress_k10")
    entry = {
        "heads": rand_keys,
        "steer_target": "context_suppress",
        "K": 10,
        "alpha_plus": ALPHA_PLUS,
        "alpha_minus": ALPHA_MINUS,
        "follows_context": r,
        "delta_pp_vs_baseline": d_pp,
        "coherence": coh,
        "label_counts": label_counts(rows),
        "bootstrap": boot["delta"],
        "flips": flips,
        "fp16_identified_suppress_delta_pp": FP16_IDENTIFIED_REF["suppress_context_k10"],
        "random_k_source": RAND_NOTE,
    }
    print(json.dumps({k: v for k, v in entry.items() if k not in ("bootstrap", "flips")}, indent=2))
    print(
        "bootstrap delta_pp",
        boot["delta"]["point_pp"],
        "CI",
        boot["delta"]["ci95_pp"],
        "excl0",
        boot["delta"]["excludes_zero"],
    )
    show_examples(rows, "random_suppress_k10")
    (OUT_DIR / "random_suppress_k10.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT_DIR / "random_k10_heads.json").write_text(
        json.dumps({"heads": rand_keys, "seed": 42, "source": RAND_NOTE}, indent=2), encoding="utf-8"
    )

    summary = {
        "model": MODEL_ID,
        "prompt_format": PROMPT_FORMAT,
        "alpha_plus": ALPHA_PLUS,
        "alpha_minus": ALPHA_MINUS,
        "n_eval": len(eval_items),
        "split_seed": split.get("seed"),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "random_k_seed": 42,
        "random_k_source": RAND_NOTE,
        "baseline": {
            "follows_context": r_base,
            "coherence": c_base,
            "label_counts": label_counts(baseline),
        },
        "conditions": {"random_suppress_k10": entry},
        "fp16_identified_suppress_reference": FP16_IDENTIFIED_REF,
    }
    summary_path = OUT_DIR / "gemma2_fp16_random_k10_suppress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    boot_payload = {
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
        "note": (
            "Resample the 100 eval items with replacement. Delta = steered "
            "follows_context − baseline follows_context, paired on item id."
        ),
        "model": MODEL_ID,
        "precision": "fp16",
        "primary_conditions": ["random_suppress_k10"],
        "random_k_source": RAND_NOTE,
        "fp16": {
            "precision": "fp16",
            "n_baseline": len(baseline),
            "conditions": {
                "random_suppress_k10": bootstrap_paired_follow_delta(
                    baseline, rows, n_boot=N_BOOT, seed=BOOT_SEED
                )
            },
        },
    }
    boot_path = OUT_DIR / "gemma2_fp16_random_k10_suppress_bootstrap.json"
    boot_path.write_text(json.dumps(boot_payload, indent=2), encoding="utf-8")
    Path("/kaggle/working/gemma2_fp16_random_k10_suppress_summary.json").write_text(
        summary_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    Path("/kaggle/working/gemma2_fp16_random_k10_suppress_bootstrap.json").write_text(
        boot_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    print("\n=== GEMMA-2-9B FP16 RANDOM-K10 SUPPRESS ===")
    hdr = f"{'condition':28s} {'fc':>6s} {'d_pp':>7s} {'CI95':>18s} {'excl0':>6s} {'coh':>6s} {'P/C/O':>8s}"
    print(hdr)
    print("-" * len(hdr))
    print(
        f"{'baseline':28s} {r_base:6.3f} {0.0:7.1f} {'—':>18s} {'—':>6s} "
        f"{c_base:6.3f} {pco(label_counts(baseline)):>8s}"
    )
    d = entry["bootstrap"]
    lo, hi = d["ci95_pp"]
    ci = f"[{lo:+.1f}, {hi:+.1f}]"
    print(
        f"{'random_suppress_k10':28s} {entry['follows_context']:6.3f} "
        f"{entry['delta_pp_vs_baseline']:7.1f} {ci:>18s} "
        f"{str(d['excludes_zero']):>6s} {entry['coherence']:6.3f} "
        f"{pco(entry['label_counts']):>8s}"
    )
    print(
        f"FP16 identified ctx K10 H− was {FP16_IDENTIFIED_REF['suppress_context_k10']:+.1f} pp "
        f"CI {FP16_IDENTIFIED_REF['ci95_pp']} excl0={FP16_IDENTIFIED_REF['excludes_zero']}"
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
