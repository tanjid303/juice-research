"""Llama-3.1-8B base GPTQ N_CALIB=128 n=100 diagnostics.

Runs only the requested parametric diagnostics:
  - noisy K2/K3 controls
  - clean-pair alpha grid
  - native INT4 K2 pair
  - raw peak L1_H4

All outputs are explicitly n=100 and use paired bootstrap n=1000, seed=42.
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

T0 = time.time()
OUT = Path("/kaggle/working/llama3_int4_n100_diagnostics")
ALPHAS = (0.5, 1.0, 2.0, 3.0, 5.0)


def elapsed(label: str) -> None:
    print(f"[elapsed] {label}: {(time.time() - T0) / 60:.1f} min")


def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [p for p in hits if p.is_file()]
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if "n100" in str(p).lower() else 1, len(str(p))))
    return hits[0]


def find_n128() -> str:
    from common.models import _looks_like_gptq_dir

    override = os.environ.get("LLAMA_INT4_CKPT")
    candidates = []
    if override:
        p = Path(override)
        if p.is_dir() and _looks_like_gptq_dir(p):
            candidates.append(p)
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if not root.exists():
            continue
        for p in root.rglob("config.json"):
            p = p.parent
            if not p.is_dir() or not _looks_like_gptq_dir(p):
                continue
            marker = str(p).replace("\\", "/").lower()
            meta = p / "quant_meta.json"
            n_calib = None
            if meta.is_file():
                try:
                    n_calib = int(json.loads(meta.read_text(encoding="utf-8")).get("n_calib"))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            if n_calib == 128 or "ncalib128" in marker:
                candidates.append(p)
    # De-duplicate paths and reject the known N_CALIB=64 directory.
    candidates = list({str(p.resolve()): p for p in candidates}.values())
    assert candidates, "Attach the existing Llama N_CALIB=128 GPTQ checkpoint"
    candidates.sort(key=lambda p: len(str(p)))
    print("N_CALIB=128 checkpoint:", candidates[0])
    return str(candidates[0])


def label_counts(rows: list[dict]) -> dict[str, int]:
    return {k: sum(r["label"] == k for r in rows) for k in ("parametric", "context", "rejects_context", "other")}


def coherent(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2 or t.count("://") >= 3 or re.fullmatch(r"[\W\d_]+", t):
        return False
    toks = t.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return False
    return sum(c.isalpha() for c in t) >= max(2, int(0.3 * len(t)))


def coherence(rows: list[dict]) -> float:
    return sum(coherent(r.get("text", "")) for r in rows) / max(1, len(rows))


def make_ident(heads, *, name: str, k: int):
    from common.juice import identified_from_enlarge

    return identified_from_enlarge(heads, precision=name, K=k, steer_target="parametric")


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "/tmp/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.51.3", "accelerate>=0.26.0,<2.0",
        "safetensors>=0.4.0", "gptqmodel==2.2.0",
        "huggingface-hub>=0.26.0", "tqdm",
    ])
    import torch
    assert torch.cuda.is_available()

    repos = list(Path("/kaggle/working").rglob("common/models.py"))
    repos += list(Path("/kaggle/input").rglob("common/models.py"))
    assert repos, "Attach the n=100 dataset ZIP"
    repo = repos[0].parent.parent
    common_dst = Path("/kaggle/working/common")
    if common_dst.exists():
        shutil.rmtree(common_dst)
    shutil.copytree(repo / "common", common_dst)
    sys.path.insert(0, "/kaggle/working")

    from common.conflictqa import load_conflict_items
    from common.inference import greedy_generate
    from common.juice import (
        HeadRef, apply_juice_steering, conflict_eval_prompt,
        context_follow_rate, head_keys, load_identified_heads,
        wrap_llama_attention_for_juice,
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

    clean_path = find_one("heads_dir_parametric_k2_clean.json")
    clean = load_identified_heads(clean_path)
    assert head_keys(clean.enlarge) == ["L20_H14", "L26_H15"]
    h = lambda s: [HeadRef(int(x.split("_H")[0][1:]), int(x.split("_H")[1])) for x in s.split(",")]
    noisy_k2 = make_ident(h("L20_H14,L31_H14"), name="noisy_parametric_k2", k=2)
    noisy_k3 = make_ident(h("L20_H14,L31_H14,L26_H15"), name="noisy_parametric_k3", k=3)
    native = make_ident(h("L16_H30,L17_H27"), name="native_int4_parametric_k2", k=2)
    raw = make_ident([HeadRef(1, 4)], name="raw_peak_l1_h4", k=1)

    model, tokenizer = load_model_and_tokenizer(
        "int4", model_id_int4=find_n128(), device_map={"": 0}
    )
    wrap_llama_attention_for_juice(model)
    elapsed("model loaded")

    def baseline_rows():
        rows = []
        for item in tqdm(items, desc="baseline"):
            text = greedy_generate(model, tokenizer, conflict_eval_prompt(tokenizer, item, prompt_format="juice_coherent"), max_new_tokens=16)
            lab = label_conflict_response(text, item.parametric_answer, item.context_answer)
            rows.append({"id": item.id, "text": text, "label": lab, "follows_context": lab == "context"})
        return rows

    baseline = baseline_rows()
    (OUT / "steer_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    base_fc = context_follow_rate(baseline)
    summary = {
        "cell": "llama_int4_n100_parametric_diagnostics",
        "model": "meta-llama/Llama-3.1-8B",
        "precision": "int4",
        "n_calib": 128,
        "n_eval": 100,
        "eval_split_seed": 42,
        "prompt_format": "juice_coherent",
        "alpha_minus": 0.0,
        "n_boot": 1000,
        "boot_seed": 42,
        "baseline": {
            "follows_context": base_fc,
            "coherence": coherence(baseline),
            "label_counts": label_counts(baseline),
        },
        "conditions": {},
    }
    boots = {}

    def run(tag, ident, alpha):
        rows = apply_juice_steering(
            model, tokenizer, items, ident, alpha_plus=alpha, alpha_minus=0.0,
            max_new_tokens=16, prompt_format="juice_coherent",
        )
        boot = bootstrap_paired_follow_delta(baseline, rows, n_boot=1000, seed=42)
        entry = {
            "heads": head_keys(ident.enlarge),
            "steer_target": "parametric",
            "K": ident.K,
            "alpha_plus": alpha,
            "alpha_minus": 0.0,
            "follows_context": context_follow_rate(rows),
            "delta_pp_vs_baseline": (context_follow_rate(rows) - base_fc) * 100,
            "coherence": coherence(rows),
            "label_counts": label_counts(rows),
            "bootstrap": boot["delta"],
        }
        summary["conditions"][tag] = entry
        boots[tag] = boot
        (OUT / f"{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(tag, json.dumps(entry, indent=2))

    run("noisy_parametric_k2", noisy_k2, 1.0)
    run("noisy_parametric_k3", noisy_k3, 1.0)
    for alpha in ALPHAS:
        run(f"clean_parametric_alpha_{alpha:g}", clean, alpha)
    run("native_parametric_k2", native, 1.0)
    run("raw_peak_l1_h4", raw, 1.0)

    (OUT / "llama_int4_n100_diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "llama_int4_n100_diagnostics_bootstrap.json").write_text(json.dumps({
        "n": 100, "n_boot": 1000, "seed": 42,
        "note": "Paired bootstrap: steered follows_context minus baseline follows_context.",
        "baseline": {"n": 100, "follows_context": base_fc},
        "conditions": {k: v for k, v in boots.items()},
    }, indent=2), encoding="utf-8")
    unload_model(model)
    elapsed("complete")


if __name__ == "__main__":
    main()
