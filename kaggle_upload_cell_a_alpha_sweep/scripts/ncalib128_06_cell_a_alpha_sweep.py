"""α-sweep for Cell A: INT4 native context K10 heads → FP16.

α+ ∈ {0.25, 0.5, 1.0}, α−=0. Reuses the FP16 Cell-3 baseline (fc 16%).
α=1 rows are reused from the existing reverse-transfer JSON if present.

No INT4 weights needed. GPU T4 ×2, HF_TOKEN, ~2–3 h if α=1 is reused, ~3–4.5 h if not.
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
TAG_PREFIX = "steer_reverse_int4_k10_fp16_a"
EXPECTED = [
    "L7_H26", "L14_H10", "L7_H21", "L4_H5", "L3_H3",
    "L31_H20", "L4_H11", "L14_H23", "L8_H29", "L9_H0",
]
ALPHAS = (0.25, 0.5, 1.0)


def elapsed(stage: str) -> None:
    print(f"[elapsed] {stage}: {(time.time() - STAGE_T0) / 60:.1f} min since start")


def _bootstrap_repo() -> Path:
    cands = list(Path("/kaggle/working").rglob("common/models.py"))
    cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
    assert cands, "Attach the reverse-k10-alpha zip (must contain common/)"

    def ok(p: Path) -> bool:
        ja = p.parent / "juice_attn.py"
        return ja.is_file() and "dense_linear_weight_T" in ja.read_text(encoding="utf-8", errors="ignore")

    cands.sort(key=lambda p: (ok(p), "alpha" in str(p).lower() or "reverse" in str(p).lower()), reverse=True)
    assert ok(cands[0])
    repo = cands[0].parent.parent
    dst = Path("/kaggle/working/common")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(repo / "common", dst)
    for p in ("/kaggle/working", str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)
    print("REPO", repo)
    return repo


def find_one(name: str, prefer: tuple[str, ...] = ()) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if any(x.lower() in str(p).lower() for x in prefer) else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]


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


def exam_meta_rate(rows) -> float:
    keys = ("multiple-choice", "test-taker", "the question is", "note:", "this question is", "the best answer is")
    n = 0
    for r in rows:
        t = (r.get("text") or "").lower()
        if any(k in t for k in keys):
            n += 1
    return n / max(1, len(rows))


def show_examples(rows, title, n_show=4):
    print(f"\n===== {title} =====")
    for r in rows[:n_show]:
        print(f"- {r['id']} [{r['label']}] {r['text']!r}")


def _reuse_alpha1() -> list[dict] | None:
    names = (
        "steer_reverse_int4_context_k10_on_fp16.json",
        "steer_reverse_int4_k10_fp16_a1.0.json",
        "steer_reverse_int4_k10_fp16_a1.json",
    )
    hits = []
    for name in names:
        try:
            hits += list(Path("/kaggle/working").rglob(name))
            if Path("/kaggle/input").exists():
                hits += list(Path("/kaggle/input").rglob(name))
        except OSError:
            pass
    hits = [h for h in hits if h.is_file()]
    if not hits:
        return None
    hits.sort(key=lambda p: len(str(p)))
    rows = json.loads(hits[0].read_text(encoding="utf-8"))
    print("reused α=1 rows ->", hits[0], "n", len(rows))
    return rows


def main() -> None:
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
            "huggingface-hub>=0.26.0",
            "tqdm",
        ]
    )
    elapsed("pip")
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

    OUT = Path("/kaggle/working/juice_stage2_int4_ncalib128")
    OUT.mkdir(parents=True, exist_ok=True)

    from common.conflictqa import load_conflict_items
    from common.juice import (
        apply_juice_steering,
        context_follow_rate,
        head_keys,
        load_identified_heads,
        wrap_llama_attention_for_juice,
    )
    from common.models import load_model_and_tokenizer, unload_model
    from common.stats_analysis import bootstrap_paired_follow_delta

    ident = load_identified_heads(find_one("heads_int4_native_context_k10.json", ("native", "directional")))
    assert head_keys(ident.enlarge) == EXPECTED
    split = json.loads(find_one("split.json", ("directional",)).read_text(encoding="utf-8"))
    data = find_one("conflictqa_pilot250_popular.jsonl")
    eval_ids = set(split["eval_ids"])
    eval_items = [it for it in load_conflict_items(data) if it.id in eval_ids]
    assert len(eval_items) == 50 and split.get("seed") == 42

    base_path = find_one("steer_baseline.json", ("fp16_reference", "base_directional"))
    baseline = json.loads(base_path.read_text(encoding="utf-8"))
    r_base = context_follow_rate(baseline)
    assert abs(r_base - 0.16) < 1e-9, r_base
    print("FP16 baseline fc", r_base, "from", base_path)

    print("Loading FP16")
    model, tokenizer = load_model_and_tokenizer("fp16", device_map="auto", forbid_cpu_offload=True)
    wrap_llama_attention_for_juice(model)
    elapsed("FP16 loaded")

    reused_a1 = _reuse_alpha1()
    conditions = {}
    boots = {}
    for a in ALPHAS:
        tag = f"{TAG_PREFIX}{a}"
        print("\n" + "=" * 70)
        print("α+", a)
        print("=" * 70)
        if a == 1.0 and reused_a1 is not None:
            rows = reused_a1
            print("using reused α=1 rows (not regenerated)")
        else:
            rows = apply_juice_steering(
                model,
                tokenizer,
                eval_items,
                ident,
                alpha_plus=a,
                alpha_minus=0.0,
                max_new_tokens=16,
                prompt_format="juice_coherent",
            )
        boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=1000, seed=42)
        r = context_follow_rate(rows)
        entry = {
            "heads": head_keys(ident.enlarge),
            "steer_target": "context",
            "K": 10,
            "alpha_plus": a,
            "alpha_minus": 0.0,
            "follows_context": r,
            "delta_pp_vs_baseline": (r - r_base) * 100,
            "coherence": coherence_rate(rows),
            "exam_meta_rate": exam_meta_rate(rows),
            "label_counts": label_counts(rows),
            "bootstrap": boot["delta"],
            "rows_reused": a == 1.0 and reused_a1 is not None,
        }
        conditions[tag] = entry
        boots[tag] = boot
        print(json.dumps({k: v for k, v in entry.items() if k != "bootstrap"}, indent=2))
        print("CI", boot["delta"]["ci95_pp"], "excl0", boot["delta"]["excludes_zero"])
        show_examples(rows, tag)
        (OUT / f"{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        elapsed(f"α={a} done")

    summary = {
        "cell": "cell_a_alpha_sweep_int4_k10_on_fp16",
        "apply_to": "fp16_base",
        "identify_on": "int4_base",
        "alphas": list(ALPHAS),
        "n_eval": 50,
        "eval_split_seed": 42,
        "n_boot": 1000,
        "boot_seed": 42,
        "baseline": {
            "follows_context": r_base,
            "coherence": coherence_rate(baseline),
            "label_counts": label_counts(baseline),
            "source": str(base_path),
        },
        "comparator_fp16_own_k10_delta_pp": 10.0,
        "reading_rule": (
            "A genuine scaled circuit should shrink toward ~+10 pp at α=0.25/0.5, "
            "keep coherence high, keep other from exploding, and drop the exam-meta dump. "
            "A stuck +40-plus with exam-meta at low α is overdrive / format collapse, not FP16's K10 object."
        ),
        "conditions": conditions,
    }
    (OUT / "cell_a_alpha_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "cell_a_alpha_sweep_bootstrap.json").write_text(
        json.dumps(
            {
                "n_boot": 1000,
                "seed": 42,
                "fp16": {"precision": "fp16", "n_baseline": 50, "conditions": boots},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== ALPHA SWEEP ===")
    print(f"{'α+':6s} {'fc':>8s} {'Δ':>8s} {'CI lo':>8s} {'CI hi':>8s} {'excl0':>7s} {'coh':>6s} {'exam':>6s} P/C/O")
    for a in ALPHAS:
        e = conditions[f"{TAG_PREFIX}{a}"]
        d = e["bootstrap"]
        lc = e["label_counts"]
        print(
            f"{a:<6} {e['follows_context']:8.3f} {e['delta_pp_vs_baseline']:8.1f} "
            f"{d['ci95_pp'][0]:8.1f} {d['ci95_pp'][1]:8.1f} {str(d['excludes_zero']):>7s} "
            f"{e['coherence']:6.3f} {e['exam_meta_rate']:6.2f} "
            f"{lc['parametric']}/{lc['context']}/{lc['other']}"
        )
    unload_model(model)
    elapsed("done")


if __name__ == "__main__":
    main()
