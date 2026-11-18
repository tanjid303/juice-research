"""FP16 context-suppress diagnostic on Gemma-2-9B base.

Steer the directional context K10 heads as H− (α+=0, α−=1, enlarge empty)
on FP16. Same 50 eval ids, juice_coherent, greedy, paired bootstrap 1000 / seed 42.

Does not load the INT4 checkpoint. Checks whether the K10 set has causal power
given the 48% FP16 baseline ceiling (enlarge at α+=1 was a null).
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

    OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_fp16_context_suppress")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ID = "google/gemma-2-9b"

    from common.conflictqa import load_conflict_items
    from common.inference import greedy_generate
    from common.juice import (
        IdentifiedHeads,
        apply_juice_steering,
        conflict_eval_prompt,
        context_follow_rate,
        head_keys,
        load_identified_heads,
    )
    from common.models import gpu_max_memory, load_model_and_tokenizer, unload_model
    from common.scoring import label_conflict_response
    from common.stats_analysis import bootstrap_paired_follow_delta
    from tqdm import tqdm

    HEADS_PATH = find_one("heads_dir_context_k10.json")
    SPLIT_PATH = find_one("split.json")
    DATA_NAME = "conflictqa_pilot250_popular.jsonl"
    data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
    data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
    assert data_hits, DATA_NAME
    DATA_PATH = data_hits[0]

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    eval_ids = set(split["eval_ids"])
    eval_items = [it for it in load_conflict_items(DATA_PATH) if it.id in eval_ids]
    assert len(eval_items) == split["n_eval"] == 50
    assert split.get("seed") == 42
    print("n_eval", len(eval_items), "seed", split.get("seed"))

    ident_ctx = load_identified_heads(HEADS_PATH)
    ident_sup = IdentifiedHeads(
        enlarge=[],
        suppress=list(ident_ctx.enlarge),
        scores_pos={},
        scores_neg={},
        precision="fp16_gemma2_base",
        K=10,
        steer_target="context",
        n_calibration=4,
        pos_alphas=[],
        neg_alphas=[],
    )
    print("suppress ctx K10", head_keys(ident_sup.suppress), "enlarge", head_keys(ident_sup.enlarge))
    assert ident_sup.enlarge == []
    assert len(ident_sup.suppress) == 10

    MAX_NEW_TOKENS = 16
    ALPHA_PLUS = 0.0
    ALPHA_MINUS = 1.0
    PROMPT_FORMAT = "juice_coherent"
    N_BOOT = 1000
    BOOT_SEED = 42

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
    print("suppress_context_k10  alpha+=0 alpha-=1  heads", head_keys(ident_sup.suppress))
    print("=" * 70)
    rows = apply_juice_steering(
        model,
        tokenizer,
        eval_items,
        ident_sup,
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
        "heads": head_keys(ident_sup.suppress),
        "steer_target": "context_suppress",
        "K": 10,
        "alpha_plus": ALPHA_PLUS,
        "alpha_minus": ALPHA_MINUS,
        "follows_context": r,
        "delta_pp_vs_baseline": d_pp,
        "coherence": coh,
        "label_counts": label_counts(rows),
        "bootstrap": boot["delta"],
        "fp16_enlarge_k10_delta_pp": -2.0,
        "note": "Enlarge of this set was −2 pp (CI includes 0). Suppress tests the opposite sign.",
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
    show_examples(rows, "suppress_context_k10")
    (OUT_DIR / "suppress_context_k10.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary = {
        "model": MODEL_ID,
        "prompt_format": PROMPT_FORMAT,
        "alpha_plus": ALPHA_PLUS,
        "alpha_minus": ALPHA_MINUS,
        "n_eval": len(eval_items),
        "split_seed": split.get("seed"),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "baseline": {
            "follows_context": r_base,
            "coherence": c_base,
            "label_counts": label_counts(baseline),
        },
        "conditions": {"suppress_context_k10": entry},
    }
    summary_path = OUT_DIR / "gemma2_fp16_context_suppress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    boot_payload = {
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
        "note": (
            "Resample the 50 eval items with replacement. Delta = steered "
            "follows_context − baseline follows_context, paired on item id."
        ),
        "model": MODEL_ID,
        "precision": "fp16",
        "primary_conditions": ["suppress_context_k10"],
        "fp16": {
            "precision": "fp16",
            "n_baseline": len(baseline),
            "conditions": {
                "suppress_context_k10": bootstrap_paired_follow_delta(
                    baseline, rows, n_boot=N_BOOT, seed=BOOT_SEED
                )
            },
        },
    }
    boot_path = OUT_DIR / "gemma2_fp16_context_suppress_bootstrap.json"
    boot_path.write_text(json.dumps(boot_payload, indent=2), encoding="utf-8")
    Path("/kaggle/working/gemma2_fp16_context_suppress_summary.json").write_text(
        summary_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    Path("/kaggle/working/gemma2_fp16_context_suppress_bootstrap.json").write_text(
        boot_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    print("\n=== GEMMA-2-9B FP16 CONTEXT SUPPRESS ===")
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
        f"{'suppress_context_k10':28s} {entry['follows_context']:6.3f} "
        f"{entry['delta_pp_vs_baseline']:7.1f} {ci:>18s} "
        f"{str(d['excludes_zero']):>6s} {entry['coherence']:6.3f} "
        f"{pco(entry['label_counts']):>8s}"
    )
    print("Enlarge of this K10 was −2 pp (null). Suppress is the opposite-sign check.")
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
