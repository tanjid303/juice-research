"""Run the two remaining Llama INT4 n=100 Figure-3 cells.

Cells:
  1. native INT4 directional context K10 enlarge
  2. FP16 directional context K10 used as INT4 H- suppression
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

OUT = Path("/kaggle/working/llama3_int4_n100_remaining")


def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    hits += list(Path("/kaggle/input").rglob(name)) if Path("/kaggle/input").exists() else []
    hits = [p for p in hits if p.is_file()]
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if "n100" in str(p).lower() else 1, len(str(p))))
    return hits[0]


def checkpoint() -> str:
    from common.models import _looks_like_gptq_dir

    found = []
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if not root.exists():
            continue
        for cfg in root.rglob("config.json"):
            p = cfg.parent
            if not _looks_like_gptq_dir(p):
                continue
            marker = str(p).lower().replace("\\", "/")
            meta = p / "quant_meta.json"
            n = None
            if meta.is_file():
                try:
                    n = int(json.loads(meta.read_text(encoding="utf-8")).get("n_calib"))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            if n == 128 or "ncalib128" in marker:
                found.append(p)
    assert found, "Attach the Llama N_CALIB=128 GPTQ checkpoint"
    found.sort(key=lambda p: len(str(p)))
    return str(found[0])


def labels(rows):
    return {k: sum(r["label"] == k for r in rows) for k in ("parametric", "context", "rejects_context", "other")}


def coh(rows):
    def ok(t):
        t = (t or "").strip()
        if len(t) < 2 or t.count("://") >= 3 or re.fullmatch(r"[\W\d_]+", t):
            return False
        return sum(c.isalpha() for c in t) >= max(2, int(0.3 * len(t)))
    return sum(ok(r.get("text", "")) for r in rows) / max(1, len(rows))


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "/tmp/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.51.3", "accelerate>=0.26.0,<2.0",
        "safetensors>=0.4.0", "gptqmodel==2.2.0",
        "huggingface-hub>=0.26.0", "tqdm",
    ])

    repos = list(Path("/kaggle/working").rglob("common/models.py"))
    repos += list(Path("/kaggle/input").rglob("common/models.py"))
    assert repos, "Attach the n=100 diagnostic dataset"
    repo = repos[0].parent.parent
    dst = Path("/kaggle/working/common")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(repo / "common", dst)
    sys.path.insert(0, "/kaggle/working")

    from common.conflictqa import load_conflict_items
    from common.inference import greedy_generate
    from common.juice import (
        apply_juice_steering, context_follow_rate, head_keys,
        load_identified_heads, identified_from_enlarge,
    )
    from common.models import load_model_and_tokenizer, unload_model
    from common.scoring import label_conflict_response
    from common.stats_analysis import bootstrap_paired_follow_delta
    from tqdm import tqdm

    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(find_one("split.json").read_text(encoding="utf-8"))
    assert split["seed"] == 42 and split["n_eval"] == 100
    data = find_one("conflictqa_pilot250_popular.jsonl")
    items = [x for x in load_conflict_items(data) if x.id in set(split["eval_ids"])]
    assert len(items) == 100

    native = load_identified_heads(find_one("heads_int4_native_context_k10.json"))
    fp16_ctx = load_identified_heads(find_one("heads_dir_context_k10.json"))
    assert len(native.enlarge) == 10 and len(fp16_ctx.enlarge) == 10

    # H- uses the FP16 context K10 heads as suppressors on INT4.
    hminus = deepcopy(fp16_ctx)
    hminus.suppress = list(hminus.enlarge)
    hminus.enlarge = []
    hminus.steer_target = "parametric"

    model, tokenizer = load_model_and_tokenizer(
        "int4", model_id_int4=checkpoint(), device_map={"": 0}
    )
    from common.juice import wrap_llama_attention_for_juice
    wrap_llama_attention_for_juice(model)

    baseline = []
    for item in tqdm(items, desc="baseline"):
        text = greedy_generate(
            model, tokenizer,
            __import__("common.juice", fromlist=["conflict_eval_prompt"]).conflict_eval_prompt(
                tokenizer, item, prompt_format="juice_coherent"
            ),
            max_new_tokens=16,
        )
        lab = label_conflict_response(text, item.parametric_answer, item.context_answer)
        baseline.append({"id": item.id, "text": text, "label": lab, "follows_context": lab == "context"})
    (OUT / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    summary = {
        "cell": "llama_int4_n100_native_and_hminus",
        "model": "meta-llama/Llama-3.1-8B",
        "precision": "int4",
        "n_calib": 128,
        "n_eval": 100,
        "eval_split_seed": 42,
        "prompt_format": "juice_coherent",
        "alpha_plus": 1.0,
        "alpha_minus": 1.0,
        "n_boot": 1000,
        "boot_seed": 42,
        "baseline": {
            "follows_context": context_follow_rate(baseline),
            "coherence": coh(baseline),
            "label_counts": labels(baseline),
        },
        "conditions": {},
    }
    boots = {}
    for tag, ident, ap, am in (
        ("native_context_k10", native, 1.0, 0.0),
        ("context_hminus_k10", hminus, 0.0, 1.0),
    ):
        rows = apply_juice_steering(
            model, tokenizer, items, ident, alpha_plus=ap, alpha_minus=am,
            max_new_tokens=16, prompt_format="juice_coherent",
        )
        boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=1000, seed=42)
        base_fc = context_follow_rate(baseline)
        entry = {
            "heads": head_keys(ident.suppress if am else ident.enlarge),
            "steer_target": "context" if ap else "parametric",
            "alpha_plus": ap,
            "alpha_minus": am,
            "follows_context": context_follow_rate(rows),
            "delta_pp_vs_baseline": (context_follow_rate(rows) - base_fc) * 100,
            "coherence": coh(rows),
            "label_counts": labels(rows),
            "bootstrap": boot["delta"],
        }
        summary["conditions"][tag] = entry
        boots[tag] = boot
        (OUT / f"{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(tag, json.dumps(entry, indent=2))

    (OUT / "llama_int4_n100_remaining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "llama_int4_n100_remaining_bootstrap.json").write_text(json.dumps({
        "n": 100, "n_boot": 1000, "seed": 42,
        "baseline": {"n": 100, "follows_context": context_follow_rate(baseline)},
        "conditions": boots,
    }, indent=2), encoding="utf-8")
    unload_model(model)
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
